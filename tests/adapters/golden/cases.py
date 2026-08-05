"""Golden corpus case registry for the adapter parity gate.

Each ``Case`` describes one deterministic adapter execution: a signature,
demos, inputs, a stub LM with EXPLICIT capability flags, optional canned
responses for multi-call flows, and which intermediate surfaces to capture.
Fixtures generated from these cases (see ``generate_fixtures.py``) pin the
adapter-to-LM boundary byte-for-byte.

Determinism rules: fixed inputs only — no network, no file reads, no
randomness, no timestamps, no environment-dependent values. Capability flags
are explicit constructor data and never derived from model names; model NAMES
are varied independently because provider-sniffing code paths (citations'
``anthropic/`` gate, reasoning's gpt-5 workaround) key on them.

Case-id convention: ``<family>--<adapter>--<slug>``. Families:
``base`` (adapter x signature-shape x demos grid), ``types`` (each special
type through each applicable adapter), ``tools`` / ``reasoning`` /
``citations`` (capability-gated planning), ``json`` (structured-output
decision tree), ``multicall`` (fallback retries, falsy double-call, TwoStep
pipelines), ``history`` (tool-call replay), ``misc`` (n, logprobs,
tool_choice passthrough), ``sink`` (kitchen-sink composites).
"""

import enum
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Literal

import pydantic

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.experimental import Citations
from dspy.signatures.field import InputField, OutputField

from .harness import Recorder, StubLM, capture_surfaces, run_adapter_capture

ADAPTER_CLASSES = {
    "chat": ChatAdapter,
    "json": JSONAdapter,
    "xml": XMLAdapter,
    "baml": BAMLAdapter,
    "two_step": TwoStepAdapter,
}

# ---------------------------------------------------------------------------
# Deterministic building blocks
# ---------------------------------------------------------------------------

IMAGE_URL = "https://example.com/golden/photo.png"
TINY_AUDIO_B64 = "c2lsZW5jZQ=="  # base64("silence")
FILE_DATA_URI = "data:text/plain;base64,aGVsbG8gZmlsZQ=="


def search_web(query: str, k: int = 3) -> str:
    """Search the web for documents."""
    return query


def lookup_user(user_id: str) -> str:
    """Look up a user profile by id."""
    return user_id


class GoldenLocation(pydantic.BaseModel):
    city: str
    country: str


class GoldenProfile(pydantic.BaseModel):
    name: str
    location: GoldenLocation
    interests: list[str]


class GoldenAnswerCard(pydantic.BaseModel):
    answer: str
    sources: list[str]


class GoldenPriority(enum.Enum):
    LOW = "low"
    HIGH = "high"


class GoldenTag(pydantic.BaseModel):
    label: str
    weight: float


class GoldenEvent(dspy.Type):
    """Custom dspy.Type exercising the content-block marker smuggling path."""

    label: str

    def format(self):
        return [{"type": "event", "event": {"label": self.label}}]

    @classmethod
    def description(cls):
        return "An event block."


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One deterministic adapter execution to pin as a fixture."""

    id: str
    adapter: str
    payload: Callable[[], dict]
    adapter_kwargs: dict = dc_field(default_factory=dict)
    lm: dict = dc_field(default_factory=dict)
    extraction_lm: dict = dc_field(default_factory=dict)
    lm_kwargs: dict = dc_field(default_factory=dict)
    responses: list = dc_field(default_factory=list)
    stop_after: int | None = None
    surfaces: tuple = ()
    modes: tuple = ("sync", "async")
    tags: tuple = ()
    #: True when the fixture's bytes legitimately depend on the python minor
    #: version (class docstrings embedded in prompt schemas: CPython 3.13+
    #: dedents docstrings at compile time, earlier versions do not). Byte
    #: authority for these cases is the pinned parity CI job.
    python_sensitive: bool = False


# ---------------------------------------------------------------------------
# Payload builders. Each returns {"signature", "demos", "inputs"} and
# optionally {"surface_outputs"} for format_finetune_data capture.
# ---------------------------------------------------------------------------


def qa_payload(demos="none"):
    class GoldenQA(dspy.Signature):
        """Answer the question briefly."""

        question: str = InputField()
        answer: str = OutputField()

    demo_sets = {
        "none": [],
        "complete": [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2 + 2?", "answer": "4"},
        ],
        # Missing the output field: exercises the incomplete-demo prefix path
        # (base.py format_demos), including its trailing-space literals.
        "incomplete": [{"question": "What is the capital of Italy?"}],
    }
    return {
        "signature": GoldenQA,
        "demos": demo_sets[demos],
        "inputs": {"question": "What is the capital of Germany?"},
        "surface_outputs": {"answer": "Berlin"},
    }


def multifield_payload(demos="none"):
    class GoldenReport(dspy.Signature):
        """Produce a careful report.

        Use every signal available and keep each field short.
        """

        context: list[str] = InputField(desc="Background passages.")
        question: str = InputField()
        verdict: Literal["yes", "no"] = OutputField(desc="Final yes/no call.")
        confidence: float = OutputField(desc="Between 0 and 1.")
        keywords: list[str] = OutputField()

    demo_sets = {
        "none": [],
        "complete": [
            {
                "context": ["The sky is blue.", "Water is wet."],
                "question": "Is the sky blue?",
                "verdict": "yes",
                "confidence": 0.9,
                "keywords": ["sky", "blue"],
            }
        ],
        "incomplete": [
            {
                "context": ["Grass is green."],
                "question": "Is grass green?",
                "verdict": "yes",
            }
        ],
    }
    return {
        "signature": GoldenReport,
        "demos": demo_sets[demos],
        "inputs": {
            "context": ["Mars is red.", "Mars has two moons."],
            "question": "Is Mars red?",
        },
        "surface_outputs": {"verdict": "yes", "confidence": 0.75, "keywords": ["mars", "red"]},
    }


def no_instructions_payload(demos="none"):
    signature = dspy.make_signature(
        "claim -> rating",
        signature_name="GoldenBareSignature",
    )
    demo_sets = {
        "none": [],
        "complete": [{"claim": "Cats are mammals.", "rating": "true"}],
        "incomplete": [{"claim": "Fish can fly."}],
    }
    return {
        "signature": signature,
        "demos": demo_sets[demos],
        "inputs": {"claim": "Birds are dinosaurs."},
        "surface_outputs": {"rating": "true"},
    }


def image_payload():
    class GoldenInspect(dspy.Signature):
        """Inspect the image."""

        image: dspy.Image = InputField()
        question: str = InputField()
        answer: str = OutputField()

    return {
        "signature": GoldenInspect,
        "demos": [
            {
                "image": dspy.Image(url=IMAGE_URL),
                "question": "What is shown?",
                "answer": "A mug.",
            }
        ],
        "inputs": {"image": dspy.Image(url=IMAGE_URL), "question": "What color is it?"},
    }


def audio_payload():
    class GoldenListen(dspy.Signature):
        """Describe the audio."""

        audio: dspy.Audio = InputField()
        answer: str = OutputField()

    return {
        "signature": GoldenListen,
        "demos": [],
        "inputs": {"audio": dspy.Audio(data=TINY_AUDIO_B64, audio_format="wav")},
    }


def file_payload():
    class GoldenFileQA(dspy.Signature):
        """Answer from the file."""

        file: dspy.File = InputField()
        question: str = InputField()
        answer: str = OutputField()

    return {
        "signature": GoldenFileQA,
        "demos": [],
        "inputs": {
            "file": dspy.File(file_data=FILE_DATA_URI, filename="notes.txt"),
            "question": "What does the file say?",
        },
    }


def document_payload():
    from dspy.adapters.types.document import Document

    class GoldenDocQA(dspy.Signature):
        """Answer with sources."""

        document: Document = InputField()
        question: str = InputField()
        answer: str = OutputField()

    return {
        "signature": GoldenDocQA,
        "demos": [],
        "inputs": {
            "document": Document(data="The Earth orbits the Sun.", title="Astronomy 101"),
            "question": "What does the Earth orbit?",
        },
    }


def code_output_payload():
    class GoldenCodeGen(dspy.Signature):
        """Write code for the task."""

        task: str = InputField()
        code: dspy.Code = OutputField()

    return {
        "signature": GoldenCodeGen,
        "demos": [{"task": "print one", "code": dspy.Code(code="print(1)")}],
        "inputs": {"task": "print hello"},
    }


def code_input_payload():
    class GoldenCodeReview(dspy.Signature):
        """Review the code."""

        code: dspy.Code = InputField()
        review: str = OutputField()

    return {
        "signature": GoldenCodeReview,
        "demos": [],
        "inputs": {"code": dspy.Code(code="def f():\n    return 42\n")},
    }


def history_payload():
    class GoldenChat(dspy.Signature):
        """Continue the conversation."""

        history: dspy.History = InputField()
        question: str = InputField()
        answer: str = OutputField()

    history = dspy.History(
        messages=[
            {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
            {"question": "When?", "answer": "Around 1600."},
        ]
    )
    return {
        "signature": GoldenChat,
        "demos": [{"history": dspy.History(messages=[]), "question": "Hi?", "answer": "Hello!"}],
        "inputs": {"history": history, "question": "Where was he born?"},
    }


def pydantic_io_payload():
    class GoldenProfileQA(dspy.Signature):
        """Answer about the profile."""

        profile: GoldenProfile = InputField()
        question: str = InputField()
        card: GoldenAnswerCard = OutputField()

    profile = GoldenProfile(
        name="Grace",
        location=GoldenLocation(city="Arlington", country="USA"),
        interests=["compilers", "navy"],
    )
    return {
        "signature": GoldenProfileQA,
        "demos": [
            {
                "profile": GoldenProfile(
                    name="Ada",
                    location=GoldenLocation(city="London", country="UK"),
                    interests=["math"],
                ),
                "question": "Who is Ada?",
                "card": GoldenAnswerCard(answer="A mathematician.", sources=["memory"]),
            }
        ],
        "inputs": {"profile": profile, "question": "Who is Grace?"},
    }


def custom_type_payload():
    class GoldenEventQA(dspy.Signature):
        """Describe the event."""

        event: GoldenEvent = InputField()
        question: str = InputField()
        answer: str = OutputField()

    return {
        "signature": GoldenEventQA,
        "demos": [],
        "inputs": {"event": GoldenEvent(label="launch"), "question": "What happened?"},
    }


def tools_payload(include_demo_tool_calls=False):
    class GoldenAgentStep(dspy.Signature):
        """Decide the next step, calling tools if useful."""

        question: str = InputField()
        tools: list[dspy.Tool] = InputField()
        next_thought: str = OutputField()
        tool_calls: dspy.ToolCalls = OutputField()

    demos = []
    if include_demo_tool_calls:
        call = dspy.ToolCalls.ToolCall(id="call_demo", name="search_web", args={"query": "owls"})
        demos = [
            {
                "question": "Find owls.",
                "tools": [dspy.Tool(search_web)],
                "next_thought": "Search for owls.",
                "tool_calls": dspy.ToolCalls(tool_calls=[call]),
            }
        ]
    return {
        "signature": GoldenAgentStep,
        "demos": demos,
        "inputs": {
            "question": "How tall is the Eiffel Tower?",
            "tools": [dspy.Tool(search_web), dspy.Tool(lookup_user)],
        },
    }


def reasoning_payload():
    class GoldenThoughtfulQA(dspy.Signature):
        """Think, then answer."""

        question: str = InputField()
        reasoning: dspy.Reasoning = OutputField()
        answer: str = OutputField()

    return {
        "signature": GoldenThoughtfulQA,
        "demos": [],
        "inputs": {"question": "Why is the sky blue?"},
    }


def citations_payload():
    from dspy.adapters.types.document import Document

    class GoldenCitedQA(dspy.Signature):
        """Answer with citations."""

        document: Document = InputField()
        question: str = InputField()
        answer: str = OutputField()
        citations: Citations = OutputField()

    return {
        "signature": GoldenCitedQA,
        "demos": [],
        "inputs": {
            "document": Document(data="Water boils at 100C.", title="Physics"),
            "question": "When does water boil?",
        },
    }


def list_of_models_payload():
    class GoldenTagging(dspy.Signature):
        """Tag the document."""

        document: str = InputField()
        prior_tags: list[GoldenTag] = InputField()
        tags: list[GoldenTag] = OutputField()

    return {
        "signature": GoldenTagging,
        "demos": [
            {
                "document": "Old doc.",
                "prior_tags": [GoldenTag(label="stale", weight=0.1)],
                "tags": [GoldenTag(label="archive", weight=0.9), GoldenTag(label="dusty", weight=0.4)],
            }
        ],
        "inputs": {
            "document": "Fresh doc about compilers.",
            "prior_tags": [GoldenTag(label="draft", weight=0.5)],
        },
    }


def dict_field_payload():
    class GoldenConfig(dspy.Signature):
        """Extract settings."""

        text: str = InputField()
        overrides: dict[str, int] = InputField()
        settings: dict[str, str] = OutputField()

    return {
        "signature": GoldenConfig,
        "demos": [],
        "inputs": {"text": "Use dark mode.", "overrides": {"retries": 3, "depth": 2}},
    }


def optional_none_payload():
    class GoldenMaybe(dspy.Signature):
        """Answer if possible."""

        question: str = InputField()
        context: str | None = InputField()
        answer: str | None = OutputField()

    return {
        "signature": GoldenMaybe,
        "demos": [{"question": "Unknowable?", "context": None, "answer": None}],
        "inputs": {"question": "What is the capital of France?", "context": None},
    }


def unicode_payload():
    class GoldenTranslate(dspy.Signature):
        """Translate."""

        text: str = InputField()
        translation: str = OutputField()

    return {
        "signature": GoldenTranslate,
        "demos": [{"text": "petit déjeuner", "translation": "早ご飯 — «breakfast» ✓"}],
        "inputs": {"text": "cœur brisé 💔 naïve"},
    }


def enum_field_payload():
    class GoldenTriage(dspy.Signature):
        """Triage the ticket."""

        ticket: str = InputField()
        priority: GoldenPriority = OutputField()

    return {
        "signature": GoldenTriage,
        "demos": [{"ticket": "Typo on docs page.", "priority": GoldenPriority.LOW}],
        "inputs": {"ticket": "Prod is down."},
    }


def open_ended_payload():
    class GoldenExtract(dspy.Signature):
        """Extract everything."""

        text: str = InputField()
        info: dict[str, Any] = OutputField()

    return {
        "signature": GoldenExtract,
        "demos": [],
        "inputs": {"text": "Alice, 30, lives in Paris."},
    }


def history_tool_replay_payload(result_id="call_1"):
    """History replay with native tool calls; result_id varies id matching.

    ``result_id='call_1'`` matches the call id; ``'call_other'`` mismatches
    (base.py's consistency guard path); ``None`` omits results entirely.
    """

    class GoldenToolHistory(dspy.Signature):
        question: str = InputField()
        history: dspy.History = InputField()
        tools: list[dspy.Tool] = InputField()
        next_thought: dspy.Reasoning = OutputField()
        tool_calls: dspy.ToolCalls = OutputField()

    tool_call = dspy.ToolCalls.ToolCall(id="call_1", name="search_web", args={"query": "cats"})
    if result_id is None:
        tool_calls = dspy.ToolCalls(tool_calls=[tool_call])
    else:
        result_call = dspy.ToolCalls.ToolCall(id=result_id, name="search_web", args={"query": "cats"})
        results = dspy.ToolCallResults.from_tool_calls_and_values([result_call], [{"items": ["cat"]}])
        tool_calls = dspy.ToolCalls(tool_calls=[tool_call], tool_call_results=results)

    history = dspy.History(
        messages=[
            {
                "question": "Q1",
                "next_thought": dspy.Reasoning(content="I should search."),
                "tool_calls": tool_calls,
            }
        ]
    )
    return {
        "signature": GoldenToolHistory,
        "demos": [],
        "inputs": {"question": "Q2", "history": history, "tools": [dspy.Tool(search_web)]},
    }


def kitchen_sink_payload():
    from dspy.adapters.types.document import Document

    class GoldenKitchenSink(dspy.Signature):
        """Answer carefully using every available signal."""

        history: dspy.History = InputField()
        image: dspy.Image = InputField()
        audio: dspy.Audio = InputField()
        file: dspy.File = InputField()
        document: Document = InputField()
        event: GoldenEvent = InputField()
        tools: list[dspy.Tool] = InputField()
        profile: GoldenProfile = InputField()
        context: str = InputField()
        question: str = InputField()
        answer: GoldenAnswerCard = OutputField()
        verdict: Literal["yes", "no"] = OutputField()
        confidence: float = OutputField()

    demo_profile = GoldenProfile(name="Ada", location=GoldenLocation(city="London", country="UK"), interests=["math"])
    history = dspy.History(
        messages=[
            {
                "profile": demo_profile,
                "context": "old note",
                "question": "Who is Ada?",
                "answer": GoldenAnswerCard(answer="A mathematician.", sources=["memory"]),
                "verdict": "yes",
                "confidence": 0.8,
            }
        ]
    )
    return {
        "signature": GoldenKitchenSink,
        "demos": [],
        "inputs": {
            "history": history,
            "image": dspy.Image(url=IMAGE_URL),
            "audio": dspy.Audio(data=TINY_AUDIO_B64, audio_format="wav"),
            "file": dspy.File(file_data=FILE_DATA_URI, filename="notes.txt"),
            "document": Document(data="The Earth orbits the Sun.", title="Astronomy"),
            "event": GoldenEvent(label="launch"),
            "tools": [dspy.Tool(search_web)],
            "profile": GoldenProfile(
                name="Grace",
                location=GoldenLocation(city="Arlington", country="USA"),
                interests=["compilers"],
            ),
            "context": "Grace joined in 2020.",
            "question": "Who is Grace?",
        },
    }


# ---------------------------------------------------------------------------
# LM capability shorthands
# ---------------------------------------------------------------------------

PLAIN_LM = {}
FC_LM = {"supports_function_calling": True}
REASONING_LM = {"supports_reasoning": True}
STRUCTURED_LM = {
    "supports_response_schema": True,
    "supported_params": ("response_format", "tools", "temperature"),
}
JSON_OBJECT_ONLY_LM = {
    "supports_response_schema": False,
    "supported_params": ("response_format",),
}


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------


def build_registry() -> dict[str, Case]:
    cases: list[Case] = []

    # -- base grid: adapter x signature shape x demo variant -----------------
    payload_builders = {
        "qa": qa_payload,
        "report": multifield_payload,
        "bare": no_instructions_payload,
    }
    for adapter in ADAPTER_CLASSES:
        for shape, builder in payload_builders.items():
            for demos in ("none", "complete", "incomplete"):
                cases.append(
                    Case(
                        id=f"base--{adapter}--{shape}-{demos}-demos",
                        adapter=adapter,
                        payload=lambda builder=builder, demos=demos: builder(demos),
                        tags=("family:base",),
                    )
                )

    # -- intermediate surfaces on the base qa case per adapter ---------------
    # BAML's format_finetune_data raising NotImplementedError is itself pinned.
    for adapter in ADAPTER_CLASSES:
        cases.append(
            Case(
                id=f"base--{adapter}--qa-surfaces",
                adapter=adapter,
                payload=lambda: qa_payload("complete"),
                surfaces=(
                    "format",
                    "format_finetune_data",
                    "format_user_message_content",
                    "format_field_with_value_probe",
                ),
                modes=("sync",),
                tags=("family:base", "surfaces"),
            )
        )

    # -- special types through every adapter ---------------------------------
    type_payloads = {
        "image": image_payload,
        "audio": audio_payload,
        "file": file_payload,
        "document": document_payload,
        "code-out": code_output_payload,
        "code-in": code_input_payload,
        "history": history_payload,
        "pydantic-io": pydantic_io_payload,
        "custom-type": custom_type_payload,
    }
    for adapter in ADAPTER_CLASSES:
        for slug, builder in type_payloads.items():
            cases.append(
                Case(
                    id=f"types--{adapter}--{slug}",
                    adapter=adapter,
                    payload=builder,
                    tags=("family:types",),
                )
            )

    # -- value-encoding shapes through every adapter (Epic B pins) ------------
    shape_payloads = {
        "list-of-models": list_of_models_payload,
        "dict-field": dict_field_payload,
        "optional-none": optional_none_payload,
        "unicode": unicode_payload,
        "enum": enum_field_payload,
    }
    for adapter in ADAPTER_CLASSES:
        for slug, builder in shape_payloads.items():
            cases.append(
                Case(
                    id=f"shapes--{adapter}--{slug}",
                    adapter=adapter,
                    payload=builder,
                    # Optional/union JSON schemas render differently on newer
                    # CPython minors; the pinned parity job is the authority.
                    python_sensitive=(slug == "optional-none"),
                    tags=("family:shapes",),
                )
            )

    # -- tools planning: native FC on/off x LM capability ---------------------
    for adapter in ("chat", "json", "xml"):
        cases.append(
            Case(
                id=f"tools--{adapter}--native-fc-supported",
                adapter=adapter,
                adapter_kwargs={"use_native_function_calling": True},
                lm=FC_LM,
                payload=tools_payload,
                tags=("family:tools",),
            )
        )
        cases.append(
            Case(
                id=f"tools--{adapter}--native-fc-unsupported-lm",
                adapter=adapter,
                adapter_kwargs={"use_native_function_calling": True},
                lm=PLAIN_LM,
                payload=tools_payload,
                tags=("family:tools",),
            )
        )
        cases.append(
            Case(
                id=f"tools--{adapter}--textual-default",
                adapter=adapter,
                payload=tools_payload,
                tags=("family:tools",),
            )
        )
    # Non-native adapters must strip provider tool params from lm_kwargs.
    cases.append(
        Case(
            id="tools--chat--kwargs-stripped-when-fc-off",
            adapter="chat",
            payload=tools_payload,
            lm_kwargs={
                "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
            tags=("family:tools",),
        )
    )
    for parallel in (True, False):
        cases.append(
            Case(
                id=f"tools--chat--parallel-tool-calls-{parallel}",
                adapter="chat",
                adapter_kwargs={
                    "use_native_function_calling": True,
                    "parallel_tool_calls": parallel,
                },
                lm=FC_LM,
                payload=tools_payload,
                tags=("family:tools",),
            )
        )
    cases.append(
        Case(
            id="tools--chat--native-with-demo-tool-calls",
            adapter="chat",
            adapter_kwargs={"use_native_function_calling": True},
            lm=FC_LM,
            payload=lambda: tools_payload(include_demo_tool_calls=True),
            tags=("family:tools",),
        )
    )
    cases.append(
        Case(
            id="misc--chat--tool-choice-passthrough",
            adapter="chat",
            adapter_kwargs={"use_native_function_calling": True},
            lm=FC_LM,
            payload=tools_payload,
            lm_kwargs={"tool_choice": {"type": "function", "function": {"name": "search_web"}}},
            tags=("family:misc", "react-v2"),
        )
    )

    # -- reasoning planning: capability x effort sourcing x model name --------
    cases.append(
        Case(
            id="reasoning--chat--native-supported",
            adapter="chat",
            lm=REASONING_LM,
            payload=reasoning_payload,
            tags=("family:reasoning",),
        )
    )
    cases.append(
        Case(
            id="reasoning--chat--textual-unsupported-lm",
            adapter="chat",
            lm=PLAIN_LM,
            payload=reasoning_payload,
            tags=("family:reasoning",),
        )
    )
    cases.append(
        Case(
            id="reasoning--chat--effort-from-lm-kwargs-attr",
            adapter="chat",
            lm={**REASONING_LM, "reasoning_effort": "high"},
            payload=reasoning_payload,
            tags=("family:reasoning", "effort-sourcing"),
        )
    )
    cases.append(
        Case(
            id="reasoning--chat--effort-from-call-kwargs",
            adapter="chat",
            lm=REASONING_LM,
            lm_kwargs={"reasoning_effort": "medium"},
            payload=reasoning_payload,
            tags=("family:reasoning", "effort-sourcing"),
        )
    )
    cases.append(
        Case(
            id="reasoning--chat--effort-explicit-none",
            adapter="chat",
            lm=REASONING_LM,
            lm_kwargs={"reasoning_effort": None},
            payload=reasoning_payload,
            tags=("family:reasoning", "effort-sourcing"),
        )
    )
    for model_type in ("chat", "responses"):
        cases.append(
            Case(
                id=f"reasoning--chat--gpt5-name-{model_type}-model-type",
                adapter="chat",
                lm={**REASONING_LM, "model": "openai/gpt-5", "model_type": model_type},
                payload=reasoning_payload,
                tags=("family:reasoning", "gpt5-workaround"),
            )
        )
    cases.append(
        Case(
            id="reasoning--json--native-supported",
            adapter="json",
            lm={**REASONING_LM, **STRUCTURED_LM, "supports_reasoning": True},
            payload=reasoning_payload,
            tags=("family:reasoning",),
        )
    )

    # -- citations: provider-name sniff x capability ---------------------------
    for model, slug in (
        ("anthropic/claude-3-7-sonnet-20250219", "anthropic-name"),
        ("openai/gpt-4o", "openai-name"),
    ):
        cases.append(
            Case(
                id=f"citations--chat--{slug}",
                adapter="chat",
                lm={"model": model},
                payload=citations_payload,
                tags=("family:citations", "provider-sniff"),
                # The Citations class docstring is embedded verbatim in the
                # rendered schema and dedents differently across pythons.
                python_sensitive=True,
            )
        )

    # -- JSONAdapter structured-output decision tree ---------------------------
    cases.append(
        Case(
            id="json--json--structured-output-model",
            adapter="json",
            lm=STRUCTURED_LM,
            payload=lambda: qa_payload("complete"),
            tags=("family:json", "structured-output"),
        )
    )
    cases.append(
        Case(
            id="json--json--no-response-format-param",
            adapter="json",
            lm=PLAIN_LM,
            payload=lambda: qa_payload("none"),
            tags=("family:json",),
        )
    )
    cases.append(
        Case(
            id="json--json--json-object-no-schema-support",
            adapter="json",
            lm=JSON_OBJECT_ONLY_LM,
            payload=lambda: qa_payload("none"),
            tags=("family:json",),
        )
    )
    cases.append(
        Case(
            id="json--json--open-ended-mapping-json-object",
            adapter="json",
            lm=STRUCTURED_LM,
            payload=open_ended_payload,
            tags=("family:json",),
        )
    )
    cases.append(
        Case(
            id="json--json--toolcalls-without-native-fc-json-object",
            adapter="json",
            lm=STRUCTURED_LM,
            payload=tools_payload,
            tags=("family:json",),
        )
    )
    cases.append(
        Case(
            id="json--baml--structured-output-model",
            adapter="baml",
            lm=STRUCTURED_LM,
            payload=pydantic_io_payload,
            tags=("family:json", "structured-output"),
        )
    )

    # -- multi-call flows -------------------------------------------------------
    # ChatAdapter -> JSONAdapter fallback retry: first response unparseable,
    # second call (JSON rendering) captured. lm_kwargs carryover is pinned
    # because the SAME dict flows into the fallback call.
    cases.append(
        Case(
            id="multicall--chat--json-fallback-retry",
            adapter="chat",
            lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[["complete garbage with no field markers"]],
            stop_after=2,
            tags=("family:multicall", "fallback"),
        )
    )
    cases.append(
        Case(
            id="multicall--chat--fallback-disabled-raises",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            responses=[["complete garbage with no field markers"]],
            stop_after=2,
            tags=("family:multicall", "fallback"),
        )
    )
    # XMLAdapter inherits the fallback (isinstance(self, JSONAdapter) guard).
    cases.append(
        Case(
            id="multicall--xml--inherited-json-fallback",
            adapter="xml",
            lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[["no xml tags at all"]],
            stop_after=2,
            tags=("family:multicall", "fallback"),
        )
    )
    # JSONAdapter falsy-result double call: the json_object branch returns an
    # empty outputs list, `if result:` is False, and the structured-output
    # path issues a SECOND call. Both payloads are pinned. NOTE the captured
    # sync/async divergence: on the async path `_json_adapter_call_common`
    # returns a coroutine (always truthy), so `if result:` short-circuits and
    # only ONE call happens. The corpus pins both behaviors bug-for-bug.
    cases.append(
        Case(
            id="multicall--json--falsy-result-double-call",
            adapter="json",
            lm=STRUCTURED_LM,
            payload=open_ended_payload,
            responses=[[]],
            stop_after=2,
            tags=("family:multicall", "falsy-double-call", "sync-async-divergence"),
        )
    )
    # TwoStep: main prose call then extraction call on the second LM.
    cases.append(
        Case(
            id="multicall--two_step--main-then-extraction",
            adapter="two_step",
            payload=lambda: qa_payload("complete"),
            responses=[["The capital of Germany is Berlin, of course."]],
            stop_after=2,
            tags=("family:multicall", "two-step", "legacy-async-quirk"),
        )
    )
    cases.append(
        Case(
            id="multicall--two_step--reasoning-output-field",
            adapter="two_step",
            extraction_lm=REASONING_LM,
            payload=reasoning_payload,
            responses=[["Because of Rayleigh scattering, obviously."]],
            stop_after=2,
            tags=("family:multicall", "two-step", "legacy-async-quirk"),
        )
    )
    # Extraction parse failure -> inner ChatAdapter falls back to JSONAdapter:
    # three LM payloads total.
    cases.append(
        Case(
            id="multicall--two_step--extraction-fallback-third-call",
            adapter="two_step",
            extraction_lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[
                ["Some long prose answer about Berlin."],
                ["garbage the extraction chat parse cannot handle"],
            ],
            stop_after=3,
            tags=("family:multicall", "two-step", "fallback", "legacy-async-quirk"),
        )
    )

    # -- history with native tool-call replay -----------------------------------
    for slug, result_id in (
        ("matching-ids", "call_1"),
        ("mismatched-ids", "call_other"),
        ("missing-results", None),
    ):
        for fc, fc_slug in ((True, "native"), (False, "textual")):
            cases.append(
                Case(
                    id=f"history--chat--tool-replay-{slug}-{fc_slug}",
                    adapter="chat",
                    adapter_kwargs={"use_native_function_calling": fc},
                    lm=FC_LM if fc else PLAIN_LM,
                    payload=lambda result_id=result_id: history_tool_replay_payload(result_id),
                    tags=("family:history", "tool-replay"),
                )
            )

    # -- misc passthroughs --------------------------------------------------------
    cases.append(
        Case(
            id="misc--chat--n-2-completions",
            adapter="chat",
            payload=lambda: qa_payload("none"),
            lm_kwargs={"n": 2},
            tags=("family:misc",),
        )
    )
    cases.append(
        Case(
            id="misc--chat--logprobs",
            adapter="chat",
            payload=lambda: qa_payload("none"),
            lm_kwargs={"logprobs": True},
            tags=("family:misc",),
        )
    )
    cases.append(
        Case(
            id="misc--chat--temperature-and-max-tokens",
            adapter="chat",
            payload=lambda: qa_payload("none"),
            lm_kwargs={"temperature": 0.3, "max_tokens": 64},
            tags=("family:misc",),
        )
    )

    # -- kitchen-sink composites ---------------------------------------------------
    for adapter in ("chat", "json"):
        cases.append(
            Case(
                id=f"sink--{adapter}--kitchen-sink",
                adapter=adapter,
                payload=kitchen_sink_payload,
                surfaces=("format",),
                tags=("family:sink",),
            )
        )

    registry = {}
    for case in cases:
        if case.id in registry:
            raise ValueError(f"Duplicate golden case id: {case.id}")
        registry[case.id] = case
    return registry


CASES = build_registry()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _build_adapter(case, recorder):
    if case.adapter == "two_step":
        extraction_lm = StubLM(recorder, name="extraction", **case.extraction_lm)
        return TwoStepAdapter(extraction_lm, **case.adapter_kwargs)
    return ADAPTER_CLASSES[case.adapter](**case.adapter_kwargs)


def execute_case(case, adapter_factory=None) -> dict:
    """Run one case in every declared mode and return its expected payload.

    A fresh recorder, stub LM, adapter, and payload are built per mode so
    adapter-side mutation (e.g. the fallback retry mutating ``lm_kwargs``,
    history deletion mutating ``inputs``) can never leak across runs.

    ``adapter_factory(case, recorder)`` lets dual-run harnesses substitute
    the adapter construction (e.g. a forced-legacy subclass) while keeping
    every other execution detail identical.
    """
    build = adapter_factory or _build_adapter
    expected = {}
    for mode in case.modes:
        recorder = Recorder(case.responses, case.stop_after)
        lm = StubLM(recorder, name="main", **case.lm)
        adapter = build(case, recorder)
        payload = case.payload()
        expected[mode] = run_adapter_capture(
            adapter,
            lm,
            case.lm_kwargs,
            payload["signature"],
            payload["demos"],
            payload["inputs"],
            mode,
            recorder,
        )
    if case.surfaces:
        recorder = Recorder()
        adapter = build(case, recorder)
        payload = case.payload()
        expected["surfaces"] = capture_surfaces(
            adapter,
            payload["signature"],
            payload["demos"],
            payload["inputs"],
            case.surfaces,
            payload.get("surface_outputs"),
        )
    return expected


def case_fixture(case) -> dict:
    """The full fixture document for one case: declaration + expectation."""
    from .harness import canonicalize

    return {
        "case_id": case.id,
        "adapter": case.adapter,
        "adapter_kwargs": canonicalize(case.adapter_kwargs),
        "lm": canonicalize(case.lm),
        "extraction_lm": canonicalize(case.extraction_lm),
        "lm_kwargs": canonicalize(case.lm_kwargs),
        "responses": canonicalize(case.responses),
        "stop_after": case.stop_after,
        "modes": list(case.modes),
        "surfaces": list(case.surfaces),
        "tags": list(case.tags),
        "expected": execute_case(case),
    }
