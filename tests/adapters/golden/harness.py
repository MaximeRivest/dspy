"""Recording harness for the adapter golden parity corpus.

This module freezes the adapter-to-LM boundary as data. A ``StubLM`` records
every call it receives (messages and kwargs, canonicalized at capture time so
later in-place mutation by adapters cannot retroactively change what was
captured) and either replays a canned response or stops the run. Fixtures
generated from these recordings pin today's adapter behavior byte-for-byte;
any future refactor must reproduce them exactly.

Design rules (see scratch/newadapter/epic-quiet-compiler.md, QC-01):

- Capability flags (``supports_function_calling`` etc.) are explicit
  constructor values. A stub NEVER consults litellm's model-capability map,
  even when its model name looks like ``anthropic/...`` or ``gpt-5...`` —
  model NAMES drive provider-sniffing code paths inside adapters, while the
  FLAGS drive capability-gated paths, and the corpus needs to vary them
  independently.
- Multi-call flows (ChatAdapter->JSONAdapter fallback retry, TwoStepAdapter's
  dual-LM pipeline, JSONAdapter's falsy-result double call) share one ordered
  ``Recorder`` across all participating stub LMs, and call COUNT is part of
  the recorded expectation.
- The run is stopped by raising ``StopGoldenCapture`` (a BaseException, so no
  adapter except-clause can swallow it) once the expected number of calls has
  been observed. This avoids hand-crafting a parseable final response for
  every signature, and keeps default-on fallback paths from firing extra
  calls after the last expected one.
"""

import asyncio
import copy
import inspect
import re

import pydantic

from dspy.clients.base_lm import BaseLM

_MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


class StopGoldenCapture(BaseException):
    """Stop adapter execution once the expected LM calls are captured.

    BaseException (not Exception) so adapter-level fallback handlers such as
    ChatAdapter's JSONAdapter retry cannot intercept it.
    """


def canonicalize(obj):
    """Convert a recorded value into a deterministic, JSON-serializable form.

    Pydantic model *classes* (JSONAdapter's structured-output
    ``response_format``) are represented structurally via ``model_json_schema()``
    so fixture comparison does not depend on object identity, per the epic's
    "compared structurally" rule.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # Reject NaN/inf at capture time: they round-trip badly through JSON.
        if obj != obj or obj in (float("inf"), float("-inf")):
            return {"__nonfinite_float__": repr(obj)}
        return obj
    if isinstance(obj, bytes):
        import base64

        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {_canonical_key(key): canonicalize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return {"__set__": sorted(str(item) for item in obj)}
    if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
        return {
            "__pydantic_model__": obj.__name__,
            "json_schema": canonicalize(obj.model_json_schema()),
        }
    if isinstance(obj, pydantic.BaseModel):
        return {
            "__pydantic_instance__": type(obj).__name__,
            "value": canonicalize(obj.model_dump()),
        }
    if isinstance(obj, type):
        return {"__type__": f"{obj.__module__}.{obj.__qualname__}"}
    return {"__repr__": _MEMORY_ADDRESS.sub(" at 0xADDR", repr(obj))}


def _canonical_key(key):
    if isinstance(key, str):
        return key
    return _MEMORY_ADDRESS.sub(" at 0xADDR", repr(key))


class Recorder:
    """Ordered, shared log of LM calls for one case execution.

    ``responses[i]`` is returned for call ``i`` (0-based) in DSPy's legacy
    outputs shape (``list[str | dict]``). Once calls run past ``stop_after``
    (default: one past the last canned response), ``StopGoldenCapture`` is
    raised after recording, so the final call's payload is still captured.
    """

    def __init__(self, responses=None, stop_after=None):
        self.responses = list(responses or [])
        if stop_after == "complete":
            # Replay-to-completion mode (parse-side corpus): never stop the
            # run; the adapter must finish and return parsed values.
            self.stop_after = None
        elif stop_after is None:
            self.stop_after = len(self.responses) + 1
        else:
            self.stop_after = stop_after
        self.calls = []

    def record(self, lm_name, messages, kwargs):
        self.calls.append(
            {
                "lm": lm_name,
                "messages": canonicalize(messages),
                "kwargs": canonicalize(kwargs),
            }
        )
        index = len(self.calls)
        if self.stop_after is not None and index >= self.stop_after:
            raise StopGoldenCapture
        if index <= len(self.responses):
            return copy.deepcopy(self.responses[index - 1])
        if self.stop_after is None:
            raise AssertionError(
                f"replay-to-completion case ran out of canned responses at call {index}; "
                "add a response for every LM call the flow makes"
            )
        raise StopGoldenCapture


class StubLM(BaseLM):
    """A BaseLM whose capabilities are explicit constructor data.

    ``**lm_attr_kwargs`` land in ``self.kwargs`` exactly like real LM
    constructor kwargs do — required by cases that pin ``reasoning_effort``
    sourcing from ``lm.kwargs`` vs call-time ``lm_kwargs``.
    """

    def __init__(
        self,
        recorder,
        name="main",
        model="stub/golden-model",
        model_type="chat",
        supports_function_calling=False,
        supports_reasoning=False,
        supports_response_schema=False,
        supported_params=(),
        **lm_attr_kwargs,
    ):
        super().__init__(model=model, model_type=model_type, cache=False, **lm_attr_kwargs)
        self._recorder = recorder
        self._name = name
        self._supports_function_calling = supports_function_calling
        self._supports_reasoning = supports_reasoning
        self._supports_response_schema = supports_response_schema
        self._supported_params = set(supported_params)

    @property
    def supports_function_calling(self):
        return self._supports_function_calling

    @property
    def supports_reasoning(self):
        return self._supports_reasoning

    @property
    def supports_response_schema(self):
        return self._supports_response_schema

    @property
    def supported_params(self):
        return set(self._supported_params)

    def __call__(self, prompt=None, messages=None, **kwargs):
        return self._recorder.record(self._name, messages, kwargs)

    async def acall(self, prompt=None, messages=None, **kwargs):
        return self._recorder.record(self._name, messages, kwargs)


def run_adapter_capture(adapter, lm, lm_kwargs, signature, demos, inputs, mode, recorder):
    """Execute one adapter call in ``mode`` ('sync' | 'async'), capturing calls.

    Returns ``{"call_count", "calls", "outcome"}`` where outcome is
    ``"stopped"`` when the capture stop fired (the normal request-side case)
    or the exception type name if the adapter raised after its final canned
    response (recorded so behavior changes in error paths are also pinned).
    """
    outcome = "completed"
    try:
        if mode == "sync":
            adapter(lm, dict(lm_kwargs), signature, list(demos), dict(inputs))
        elif mode == "async":
            asyncio.run(adapter.acall(lm, dict(lm_kwargs), signature, list(demos), dict(inputs)))
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    except StopGoldenCapture:
        outcome = "stopped"
    except Exception as error:
        outcome = f"raised:{type(error).__name__}"
    return {
        "call_count": len(recorder.calls),
        "calls": recorder.calls,
        "outcome": outcome,
    }


def run_adapter_to_completion(adapter, lm, lm_kwargs, signature, demos, inputs, mode, recorder):
    """Execute one adapter call to completion, capturing parsed values.

    Unlike ``run_adapter_capture`` (request-side: stops at the last expected
    call), this replays every canned response and records what the adapter
    PARSED: the returned value dicts with their typed objects canonicalized,
    or the exception it raised. Recorded LM payloads are kept so fallback
    flows pin all calls alongside the final result.
    """
    result = {}
    try:
        if mode == "sync":
            values = adapter(lm, dict(lm_kwargs), signature, list(demos), dict(inputs))
        elif mode == "async":
            values = asyncio.run(adapter.acall(lm, dict(lm_kwargs), signature, list(demos), dict(inputs)))
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        result["outcome"] = "completed"
        result["values"] = canonicalize(values)
    except Exception as error:
        result["outcome"] = f"raised:{type(error).__name__}"
        result["error"] = canonical_error(error)
    result["call_count"] = len(recorder.calls)
    result["calls"] = recorder.calls
    return result


def canonical_error(error):
    """Pin an exception as fixture data.

    ``AdapterParseError`` is dspy-owned, so its message and attributes are
    pinned exactly; other exception types pin only the type name because
    their wording may belong to third-party libraries.
    """
    from dspy.utils.exceptions import AdapterParseError

    info = {"type": type(error).__name__}
    if isinstance(error, AdapterParseError):
        info["message"] = canonicalize(str(error))
        info["adapter_name"] = canonicalize(getattr(error, "adapter_name", None))
        info["lm_response"] = canonicalize(getattr(error, "lm_response", None))
        info["parsed_result"] = canonicalize(getattr(error, "parsed_result", None))
    return info


class CallbackProbe:
    """Records the ordered adapter callback event stream as strings.

    Entries look like ``"format_start:ChatAdapter"`` / ``"format_end"`` /
    ``"parse_end:raised:AdapterParseError"``. End hooks do not receive the
    instance, so pairing is positional — which is exactly what the fixture
    pins (including the JSONAdapter/XMLAdapter double-fire from depth-2
    ``with_callbacks`` wrapping and TwoStep's nested inner-adapter events).
    Stub LMs override ``__call__`` without ``with_callbacks``, so no LM
    events appear; the corpus pins ADAPTER event sequences only.
    """

    def __init__(self):
        from dspy.utils.callback import BaseCallback

        # Compose dynamically so this module never subclasses at import time
        # with a stale signature if BaseCallback gains hooks.
        probe = self

        class _Probe(BaseCallback):
            def on_adapter_format_start(self, call_id, instance, inputs):
                probe.events.append(f"format_start:{type(instance).__name__}")

            def on_adapter_format_end(self, call_id, outputs, exception):
                probe.events.append(_end_event("format_end", exception))

            def on_adapter_parse_start(self, call_id, instance, inputs):
                probe.events.append(f"parse_start:{type(instance).__name__}")

            def on_adapter_parse_end(self, call_id, outputs, exception):
                probe.events.append(_end_event("parse_end", exception))

        self.events = []
        self.callback = _Probe()


def _end_event(name, exception):
    if exception is None:
        return name
    return f"{name}:raised:{type(exception).__name__}"


def capture_surfaces(adapter, signature, demos, inputs, surfaces, surface_outputs=None):
    """Capture intermediate render surfaces consumed below the LM boundary.

    These are parity surfaces because optimizers and modules read them
    directly: ``adapter.format(...)`` (GRPO, GEPA), ``format_finetune_data``
    (bootstrap_finetune), ``format_user_message_content`` (ReAct trajectories),
    and the ``format_field_with_value`` arity/behavior probe (DummyLM).
    """
    captured = {}
    for surface in surfaces:
        try:
            if surface == "format":
                value = adapter.format(signature, list(demos), dict(inputs))
            elif surface == "format_finetune_data":
                value = adapter.format_finetune_data(signature, list(demos), dict(inputs), dict(surface_outputs or {}))
            elif surface == "format_user_message_content":
                value = adapter.format_user_message_content(signature, dict(inputs))
            elif surface == "format_field_with_value_probe":
                value = _field_with_value_probe(adapter)
            else:
                raise ValueError(f"Unknown surface: {surface!r}")
            captured[surface] = canonicalize(value)
        except Exception as error:
            captured[surface] = {"__raises__": type(error).__name__}
    return captured


def _field_with_value_probe(adapter):
    """Pin both the signature (arity) and output of format_field_with_value.

    DummyLM (dspy/utils/dummies.py:111-125) calls this method to render canned
    answers, and JSONAdapter's variant takes an extra ``role`` parameter — the
    probe makes any signature or rendering change a corpus diff.
    """
    from dspy.adapters.utils import FieldInfoWithName
    from dspy.signatures.field import OutputField

    parameters = list(inspect.signature(adapter.format_field_with_value).parameters)
    field = FieldInfoWithName(name="probe_field", info=OutputField())
    rendered = adapter.format_field_with_value({field: "probe value"})
    return {"parameters": parameters, "rendered": rendered}
