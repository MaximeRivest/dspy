"""The parser-hook contract and the response-view facade.

The hook signature ``parse(response_view, ctx) -> dict`` is frozen from this
PR onward: when the engine later consumes typed ``LMResponse`` objects
directly (closing base.py's TODO #3), only the facade's backing swaps — no
parser is rewritten. Future leniency/repair knobs attach via ``ctx``, never
by re-signaturing hooks.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ResponseView:
    """A uniform read surface over one LM output.

    Today this wraps DSPy's legacy output shape (``str`` or
    ``{"text", "logprobs", "tool_calls", ...}``). After the LMResponse
    cutover it will wrap a typed output; consumers only ever use this
    surface, which is why the swap is free.
    """

    def __init__(self, legacy_output: Any):
        self._output = legacy_output

    @classmethod
    def from_lm_response(cls, response) -> list["ResponseView"]:
        """Views over a typed LMResponse — the facade swap that closes
        TODO #3 for the engine path.

        Backing payloads are derived by the SAME canonical rules the legacy
        path uses (``legacy_outputs_from_lm_response``: provider_output
        verbatim, text-only as str, else the output dict), so str-vs-dict
        provider fidelity is shared code, not a reimplementation.
        """
        from dspy.clients.openai_format import legacy_outputs_from_lm_response

        return [cls(output) for output in legacy_outputs_from_lm_response(response)]

    @property
    def text(self) -> str | None:
        if isinstance(self._output, dict):
            return self._output.get("text")
        return self._output

    @property
    def tool_calls(self) -> Any:
        if isinstance(self._output, dict):
            return self._output.get("tool_calls")
        return None

    @property
    def logprobs(self) -> Any:
        if isinstance(self._output, dict):
            return self._output.get("logprobs")
        return None

    def channel(self, name: str) -> Any:
        """Read a provider/native channel (``reasoning_content``,
        ``citations``, ...) if present."""
        if isinstance(self._output, dict):
            return self._output.get(name)
        return None

    @property
    def raw(self) -> Any:
        """Escape hatch for engine-internal compatibility shims only."""
        return self._output


@dataclass
class ParseContext:
    """Everything a parser hook may consult while parsing one output.

    ``lm`` is included because some parsers legitimately call a model —
    TwoStepAdapter's extraction stage is a plan-carried parser that invokes
    its extraction LM.
    """

    plan: Any
    signature: Any = None
    lm: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_fields(self):
        return self.plan.output_fields if self.plan is not None else []


@runtime_checkable
class ParserHook(Protocol):
    """A response-to-values hook contributed by a format or strategy."""

    name: str

    def parse(self, response_view: ResponseView, ctx: ParseContext) -> dict[str, Any]: ...


class ThirdPartyNativeParserHook:
    """Silent compatibility hook: a third-party Type in
    ``native_response_types`` keeps its documented ``parse_lm_response``
    contract on the engine path (receiving the legacy-shaped payload via the
    view's raw backing)."""

    name = "type_hook.parse_lm_response"

    def __init__(self, annotation, destination: str):
        self.annotation = annotation
        self.destination = destination

    def parse(self, response_view: ResponseView, ctx: ParseContext) -> dict[str, Any]:
        parsed_value = self.annotation.parse_lm_response(response_view.raw)
        if parsed_value is not None:
            return {self.destination: parsed_value}
        return {}
