"""Parse-side and callback-sequence golden cases.

Where ``cases.py`` pins what adapters SEND (request side), this module pins
what they RETURN: parsed value dicts with typed objects, exception semantics
(type, message, attributes for dspy-owned ``AdapterParseError``), fallback
chains with every LM payload, and the ordered adapter-callback event streams.

Replay boundaries are explicit per case via ``runners``:

- ``"call"`` / ``"acall"``: the full adapter pipeline with a
  replay-to-completion recorder (canned responses for EVERY call the flow
  makes — fallback retries and TwoStep extraction included).
- ``"parse"``: direct ``adapter.parse(signature, text)`` — pure text-parse
  semantics with no preprocessing, fallback, or native-type handling.
- ``"postprocess"``: direct ``adapter._call_postprocess(...)`` — pins the
  method signature relied on by existing tests and the native tool-call
  merge behavior.

Known divergences pinned bug-for-bug carry tags: ``legacy-async-quirk``
(TwoStep's async path differs from sync in four verified ways) and
``sync-async-divergence`` (JSONAdapter's coroutine-truthiness short-circuit).
"""

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Callable

from .cases import (
    ADAPTER_CLASSES,
    REASONING_LM,
    STRUCTURED_LM,
    citations_payload,
    code_output_payload,
    enum_field_payload,
    list_of_models_payload,
    multifield_payload,
    optional_none_payload,
    pydantic_io_payload,
    qa_payload,
    tools_payload,
    unicode_payload,
)
from .harness import (
    CallbackProbe,
    Recorder,
    StubLM,
    canonical_error,
    canonicalize,
    run_adapter_to_completion,
)

CHAT_QA_OK = "[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]"
JSON_QA_OK = '{"answer": "Berlin"}'
XML_QA_OK = "<answer>Berlin</answer>"
GARBAGE = "complete garbage with no field markers"

OPENAI_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "search_web", "arguments": '{"query": "eiffel tower height"}'},
}


@dataclass(frozen=True)
class ParseCase:
    """One deterministic parse-side execution to pin as a fixture."""

    id: str
    adapter: str
    payload: Callable[[], dict]
    adapter_kwargs: dict = dc_field(default_factory=dict)
    lm: dict = dc_field(default_factory=dict)
    extraction_lm: dict = dc_field(default_factory=dict)
    lm_kwargs: dict = dc_field(default_factory=dict)
    responses: list = dc_field(default_factory=list)
    runners: tuple = ("call", "acall")
    parse_text: str | None = None
    postprocess_output: object = None
    tags: tuple = ()
    #: See Case.python_sensitive: recorded payloads embed a class docstring
    #: whose dedent differs across python minors (CPython 3.13+).
    python_sensitive: bool = False


def build_parse_registry() -> dict[str, ParseCase]:
    cases: list[ParseCase] = []

    # -- happy paths per adapter, full pipeline AND direct parse --------------
    happy = {
        "chat": CHAT_QA_OK,
        "json": JSON_QA_OK,
        "xml": XML_QA_OK,
        "baml": JSON_QA_OK,
    }
    for adapter, response in happy.items():
        cases.append(
            ParseCase(
                id=f"parse--{adapter}--qa-happy",
                adapter=adapter,
                adapter_kwargs={"use_json_adapter_fallback": False} if adapter in ("chat", "xml") else {},
                payload=lambda: qa_payload("none"),
                responses=[[response]],
                runners=("call", "acall", "parse"),
                parse_text=response,
                tags=("family:parse",),
            )
        )
    cases.append(
        ParseCase(
            id="parse--two_step--qa-happy",
            adapter="two_step",
            payload=lambda: qa_payload("none"),
            responses=[["The capital is Berlin, naturally."], [CHAT_QA_OK]],
            tags=("family:parse", "two-step"),
        )
    )

    # JSON wrapped in a fenced block and in prose: the recursive-regex /
    # json-repair extraction path.
    cases.append(
        ParseCase(
            id="parse--json--fenced-block",
            adapter="json",
            payload=lambda: qa_payload("none"),
            responses=[['```json\n{"answer": "Berlin"}\n```']],
            runners=("call", "acall", "parse"),
            parse_text='```json\n{"answer": "Berlin"}\n```',
            tags=("family:parse",),
        )
    )
    cases.append(
        ParseCase(
            id="parse--json--prose-wrapped",
            adapter="json",
            payload=lambda: qa_payload("none"),
            responses=[['Sure! Here is the JSON you asked for: {"answer": "Berlin"} Hope this helps.']],
            runners=("call", "acall", "parse"),
            parse_text='Sure! Here is the JSON you asked for: {"answer": "Berlin"} Hope this helps.',
            tags=("family:parse",),
        )
    )

    # -- typed coercion -------------------------------------------------------
    report_chat = (
        "[[ ## verdict ## ]]\nyes\n\n[[ ## confidence ## ]]\n0.9\n\n"
        '[[ ## keywords ## ]]\n["mars", "red"]\n\n[[ ## completed ## ]]'
    )
    cases.append(
        ParseCase(
            id="parse--chat--typed-fields",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: multifield_payload("none"),
            responses=[[report_chat]],
            runners=("call", "acall", "parse"),
            parse_text=report_chat,
            tags=("family:parse", "coercion"),
        )
    )
    report_json = '{"verdict": "yes", "confidence": 0.9, "keywords": ["mars", "red"]}'
    cases.append(
        ParseCase(
            id="parse--json--typed-fields",
            adapter="json",
            payload=lambda: multifield_payload("none"),
            responses=[[report_json]],
            runners=("call", "acall", "parse"),
            parse_text=report_json,
            tags=("family:parse", "coercion"),
        )
    )
    # Numeric underscore: json_repair vs ast.literal_eval fallback inside
    # parse_value. Pins whatever today's chain produces.
    underscore = (
        "[[ ## verdict ## ]]\nyes\n\n[[ ## confidence ## ]]\n1_000.25\n\n"
        '[[ ## keywords ## ]]\n["x"]\n\n[[ ## completed ## ]]'
    )
    cases.append(
        ParseCase(
            id="parse--chat--float-underscore-edge",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: multifield_payload("none"),
            responses=[[underscore]],
            runners=("call", "parse"),
            parse_text=underscore,
            tags=("family:parse", "json-repair-edge"),
        )
    )
    card_chat = '[[ ## card ## ]]\n{"answer": "Grace is a programmer.", "sources": ["bio"]}\n\n[[ ## completed ## ]]'
    cases.append(
        ParseCase(
            id="parse--chat--pydantic-output",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=pydantic_io_payload,
            responses=[[card_chat]],
            runners=("call", "parse"),
            parse_text=card_chat,
            tags=("family:parse", "coercion"),
        )
    )
    # -- value-encoding shape coercion (Epic B pins) --------------------------
    tags_chat = (
        "[[ ## tags ## ]]\n"
        '[{"label": "compilers", "weight": 0.8}, {"label": "systems", "weight": 0.3}]\n\n'
        "[[ ## completed ## ]]"
    )
    cases.append(
        ParseCase(
            id="parse--chat--list-of-models",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=list_of_models_payload,
            responses=[[tags_chat]],
            runners=("call", "parse"),
            parse_text=tags_chat,
            tags=("family:parse", "coercion", "shapes"),
        )
    )
    tags_json = '{"tags": [{"label": "compilers", "weight": 0.8}, {"label": "systems", "weight": 0.3}]}'
    cases.append(
        ParseCase(
            id="parse--json--list-of-models",
            adapter="json",
            payload=list_of_models_payload,
            responses=[[tags_json]],
            runners=("call", "parse"),
            parse_text=tags_json,
            tags=("family:parse", "coercion", "shapes"),
        )
    )
    enum_chat = "[[ ## priority ## ]]\nhigh\n\n[[ ## completed ## ]]"
    cases.append(
        ParseCase(
            id="parse--chat--enum-by-value",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=enum_field_payload,
            responses=[[enum_chat]],
            runners=("call", "parse"),
            parse_text=enum_chat,
            tags=("family:parse", "coercion", "shapes"),
        )
    )
    optional_chat = "[[ ## answer ## ]]\nnull\n\n[[ ## completed ## ]]"
    cases.append(
        ParseCase(
            id="parse--chat--optional-null",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=optional_none_payload,
            responses=[[optional_chat]],
            runners=("call", "parse"),
            parse_text=optional_chat,
            python_sensitive=True,
            tags=("family:parse", "coercion", "shapes"),
        )
    )
    unicode_chat = "[[ ## translation ## ]]\ncœur brisé 💔 — «naïve» ✓\n\n[[ ## completed ## ]]"
    cases.append(
        ParseCase(
            id="parse--chat--unicode-passthrough",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=unicode_payload,
            responses=[[unicode_chat]],
            runners=("call", "parse"),
            parse_text=unicode_chat,
            tags=("family:parse", "coercion", "shapes"),
        )
    )

    fenced_code = "[[ ## code ## ]]\n```python\nprint(1)\n```\n\n[[ ## completed ## ]]"
    cases.append(
        ParseCase(
            id="parse--chat--code-fence-stripping",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=code_output_payload,
            responses=[[fenced_code]],
            runners=("call", "parse"),
            parse_text=fenced_code,
            tags=("family:parse", "dspy-type-retry"),
        )
    )

    # -- native response channels ----------------------------------------------
    cases.append(
        ParseCase(
            id="parse--chat--native-reasoning-content",
            adapter="chat",
            lm=REASONING_LM,
            payload=qa_payload_with_reasoning,
            responses=[[{"text": CHAT_QA_OK, "reasoning_content": "I considered the geography."}]],
            tags=("family:parse", "native-reasoning"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--native-citations",
            adapter="chat",
            lm={"model": "anthropic/claude-3-7-sonnet-20250219"},
            payload=citations_payload,
            python_sensitive=True,
            responses=[
                [
                    {
                        "text": "[[ ## answer ## ]]\nAt 100C.\n\n[[ ## completed ## ]]",
                        "citations": [
                            {
                                "cited_text": "Water boils at 100C.",
                                "document_index": 0,
                                "start_char_index": 0,
                                "end_char_index": 20,
                            }
                        ],
                    }
                ]
            ],
            tags=("family:parse", "native-citations", "provider-sniff"),
        )
    )
    # Native tool calls: AdapterParseError on the text is TOLERATED when
    # tool_calls are present (base.py), and provider-shaped calls are merged.
    cases.append(
        ParseCase(
            id="parse--chat--native-tool-calls-tolerant",
            adapter="chat",
            adapter_kwargs={"use_native_function_calling": True},
            lm={"supports_function_calling": True},
            payload=tools_payload,
            responses=[[{"text": "", "tool_calls": [OPENAI_TOOL_CALL]}]],
            tags=("family:parse", "native-tools"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--postprocess-direct-tool-calls",
            adapter="chat",
            adapter_kwargs={"use_native_function_calling": True},
            lm={"supports_function_calling": True},
            payload=tools_payload,
            runners=("postprocess",),
            postprocess_output={"text": "", "tool_calls": [OPENAI_TOOL_CALL]},
            tags=("family:parse", "native-tools", "pins-postprocess-signature"),
        )
    )

    # -- error and fallback semantics -------------------------------------------
    cases.append(
        ParseCase(
            id="parse--chat--garbage-no-fallback-raises",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            responses=[[GARBAGE]],
            runners=("call", "acall", "parse"),
            parse_text=GARBAGE,
            tags=("family:parse", "errors"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--fallback-success-second-call",
            adapter="chat",
            lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[[GARBAGE], ['{"answer": "Paris"}']],
            tags=("family:parse", "fallback"),
        )
    )
    # The full failure chain is THREE calls: Chat parse fails -> fallback
    # JSONAdapter's structured-output attempt -> its parse failure is caught
    # by the generic `except Exception` in JSONAdapter.__call__ (logging
    # "Failed to use structured output format") -> json_object retry -> final
    # AdapterParseError propagates.
    cases.append(
        ParseCase(
            id="parse--chat--fallback-both-fail-reraise",
            adapter="chat",
            lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[[GARBAGE], [GARBAGE], [GARBAGE]],
            tags=("family:parse", "fallback", "errors"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--empty-text-raises",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            responses=[[""]],
            tags=("family:parse", "errors"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--empty-outputs-list",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            responses=[[]],
            tags=("family:parse", "errors", "empty-outputs"),
        )
    )

    # -- multiple completions and logprobs ----------------------------------------
    cases.append(
        ParseCase(
            id="parse--chat--n-2-two-value-dicts",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            lm_kwargs={"n": 2},
            responses=[[CHAT_QA_OK, "[[ ## answer ## ]]\nMunich\n\n[[ ## completed ## ]]"]],
            tags=("family:parse",),
        )
    )
    cases.append(
        ParseCase(
            id="parse--chat--logprobs-attached",
            adapter="chat",
            adapter_kwargs={"use_json_adapter_fallback": False},
            payload=lambda: qa_payload("none"),
            lm_kwargs={"logprobs": True},
            responses=[[{"text": CHAT_QA_OK, "logprobs": {"content": [{"token": "Berlin", "logprob": -0.1}]}}]],
            tags=("family:parse",),
        )
    )

    # -- TwoStep async quirks (sync and async pinned separately) -------------------
    cases.append(
        ParseCase(
            id="parse--two_step--empty-main-text-sync",
            adapter="two_step",
            payload=lambda: qa_payload("none"),
            responses=[[""]],
            runners=("call",),
            tags=("family:parse", "two-step", "legacy-async-quirk", "errors"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--two_step--empty-main-text-async",
            adapter="two_step",
            payload=lambda: qa_payload("none"),
            # The async path invokes extraction even on empty main text — the
            # second canned response feeds that extraction call.
            responses=[[""], [CHAT_QA_OK]],
            runners=("acall",),
            tags=("family:parse", "two-step", "legacy-async-quirk"),
        )
    )
    cases.append(
        ParseCase(
            id="parse--two_step--extraction-fallback-result",
            adapter="two_step",
            extraction_lm=STRUCTURED_LM,
            payload=lambda: qa_payload("none"),
            responses=[["Berlin is the answer, clearly."], [GARBAGE], ['{"answer": "Berlin"}']],
            tags=("family:parse", "two-step", "fallback"),
        )
    )

    registry = {}
    for case in cases:
        if case.id in registry:
            raise ValueError(f"Duplicate parse case id: {case.id}")
        registry[case.id] = case
    return registry


def qa_payload_with_reasoning():
    from .cases import reasoning_payload

    return reasoning_payload()


PARSE_CASES = build_parse_registry()


# ---------------------------------------------------------------------------
# Callback-sequence cases: representative flows whose adapter event streams
# are pinned (sync AND async), including the JSONAdapter/XMLAdapter
# double-fire from depth-2 with_callbacks wrapping, TwoStep's nested
# inner-ChatAdapter events, and any async absence of parse events.
# ---------------------------------------------------------------------------

CALLBACK_CASE_IDS = (
    "parse--chat--qa-happy",
    "parse--json--qa-happy",
    "parse--xml--qa-happy",
    "parse--two_step--qa-happy",
    "parse--chat--fallback-success-second-call",
    "parse--chat--garbage-no-fallback-raises",
)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _build(case, recorder, callbacks=None):
    from .cases import TwoStepAdapter  # re-exported import for clarity

    lm = StubLM(recorder, name="main", **case.lm)
    adapter_kwargs = dict(case.adapter_kwargs)
    if callbacks is not None:
        adapter_kwargs["callbacks"] = callbacks
    if case.adapter == "two_step":
        extraction_lm = StubLM(recorder, name="extraction", **case.extraction_lm)
        adapter = TwoStepAdapter(extraction_lm, **adapter_kwargs)
    else:
        adapter = ADAPTER_CLASSES[case.adapter](**adapter_kwargs)
    return adapter, lm


def execute_parse_case(case) -> dict:
    expected = {}
    for runner in case.runners:
        if runner in ("call", "acall"):
            recorder = Recorder(case.responses, stop_after="complete")
            adapter, lm = _build(case, recorder)
            payload = case.payload()
            mode = "sync" if runner == "call" else "async"
            expected[runner] = run_adapter_to_completion(
                adapter, lm, case.lm_kwargs, payload["signature"], payload["demos"], payload["inputs"], mode, recorder
            )
        elif runner == "parse":
            adapter, _ = _build(case, Recorder())
            payload = case.payload()
            try:
                values = adapter.parse(payload["signature"], case.parse_text)
                expected[runner] = {"outcome": "completed", "values": canonicalize(values)}
            except Exception as error:
                expected[runner] = {
                    "outcome": f"raised:{type(error).__name__}",
                    "error": canonical_error(error),
                }
        elif runner == "postprocess":
            recorder = Recorder()
            adapter, lm = _build(case, recorder)
            payload = case.payload()
            signature = payload["signature"]
            # Reproduce preprocessing's field deletion for the processed
            # signature exactly as Adapter.__call__ would before parsing.
            processed = adapter._call_preprocess(lm, {}, signature, dict(payload["inputs"]))
            try:
                values = adapter._call_postprocess(processed, signature, [case.postprocess_output], lm, {})
                expected[runner] = {"outcome": "completed", "values": canonicalize(values)}
            except Exception as error:
                expected[runner] = {
                    "outcome": f"raised:{type(error).__name__}",
                    "error": canonical_error(error),
                }
        else:
            raise ValueError(f"Unknown runner: {runner!r}")
    return expected


def execute_callback_case(case) -> dict:
    expected = {}
    for runner in ("call", "acall"):
        probe = CallbackProbe()
        recorder = Recorder(case.responses, stop_after="complete")
        adapter, lm = _build(case, recorder, callbacks=[probe.callback])
        payload = case.payload()
        mode = "sync" if runner == "call" else "async"
        result = run_adapter_to_completion(
            adapter, lm, case.lm_kwargs, payload["signature"], payload["demos"], payload["inputs"], mode, recorder
        )
        expected[runner] = {"outcome": result["outcome"], "events": list(probe.events)}
    return expected


def parse_case_fixture(case) -> dict:
    return {
        "case_id": case.id,
        "adapter": case.adapter,
        "adapter_kwargs": canonicalize(case.adapter_kwargs),
        "lm": canonicalize(case.lm),
        "extraction_lm": canonicalize(case.extraction_lm),
        "lm_kwargs": canonicalize(case.lm_kwargs),
        "responses": canonicalize(case.responses),
        "runners": list(case.runners),
        "parse_text": case.parse_text,
        "postprocess_output": canonicalize(case.postprocess_output),
        "tags": list(case.tags),
        "expected": execute_parse_case(case),
    }


def callback_case_fixture(case) -> dict:
    return {
        "case_id": case.id,
        "adapter": case.adapter,
        "tags": list(case.tags),
        "expected": execute_callback_case(case),
    }
