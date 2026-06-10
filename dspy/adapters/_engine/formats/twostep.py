"""TwoStepFormat: TwoStepAdapter's prose literals, frozen.

TwoStep's message pipeline differs STRUCTURALLY from the base adapters (no
field-structure section, no history branch, no output-requirements request,
no custom-type marker expansion), so its assembly lives here as
``render_two_step_messages`` — deliberately NOT in ``render.py``, whose
purity contract (zero format-specific structure) stays intact permanently.

The extraction stage is recorded on plans as a parser hook that carries the
extraction model: representation decisions travel with their parsers, even
when the "parser" is itself an LM call.
"""

from typing import Any

from dspy.adapters._engine.formats import Format
from dspy.adapters._engine.render import _render_demos
from dspy.adapters.utils import get_field_description_string


class TwoStepFormat(Format):
    def render_system(self, signature) -> str:
        return self.render_task_description(signature)

    def render_task_description(self, signature) -> str:
        parts = []

        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
        parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

        if signature.instructions:
            parts.append(f"Specific instructions: {signature.instructions}")

        return "\n".join(parts)

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        # Legacy TwoStep has no main_request parameter at all; the kwarg is
        # accepted for interface compatibility and deliberately ignored.
        parts = [prefix]

        for name in signature.input_fields.keys():
            if name in inputs:
                parts.append(f"{name}: {inputs.get(name, '')}")

        parts.append(suffix)
        return "\n\n".join(parts).strip()

    def render_assistant_content(self, signature, outputs: dict[str, Any], missing_field_message=None) -> str:
        parts = []

        for name in signature.output_fields.keys():
            if name in outputs:
                parts.append(f"{name}: {outputs.get(name, missing_field_message)}")

        return "\n\n".join(parts).strip()

    def make_parser_hook(self, adapter):
        return TwoStepExtractionParserHook(adapter.extraction_model, adapter._create_extractor_signature)


def render_two_step_messages(fmt: TwoStepFormat, signature, demos: list[dict[str, Any]], inputs: dict[str, Any]):
    """Structural twin of legacy ``TwoStepAdapter.format``: prose system
    message, base demo classification, plain user content — and none of the
    base pipeline's history handling or marker expansion."""
    messages = []
    messages.append({"role": "system", "content": fmt.render_system(signature)})
    messages.extend(_render_demos(fmt, signature, demos))
    messages.append({"role": "user", "content": fmt.render_user_content(signature, inputs)})
    return messages


class TwoStepExtractionParserHook:
    """The plan-carried record of TwoStep's extraction stage.

    The extraction "parser" is an LM call: a bare ChatAdapter run against
    the extraction model over a synthetic ``text -> outputs`` signature.
    Recorded so engine plans are complete; the adapter's ``parse`` remains
    the executing implementation.
    """

    name = "two_step.extraction"

    def __init__(self, extraction_model, create_extractor_signature):
        self.extraction_model = extraction_model
        self._create_extractor_signature = create_extractor_signature

    def parse(self, response_view, ctx) -> dict[str, Any]:
        # Error identity mirrors legacy TwoStepAdapter.parse exactly: LM
        # errors propagate; anything else wraps as TwoStepAdapter's
        # AdapterParseError.
        from dspy.adapters.chat_adapter import ChatAdapter
        from dspy.utils.exceptions import AdapterParseError, LMError

        extractor_signature = self._create_extractor_signature(ctx.signature)
        try:
            return ChatAdapter()(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": response_view.text},
            )[0]
        except LMError:
            raise
        except Exception as e:
            raise AdapterParseError(
                adapter_name="TwoStepAdapter",
                signature=ctx.signature,
                lm_response=response_view.text,
                message=f"Failed to parse response from the original completion: {e}",
            ) from e
