import re
from typing import Any, NamedTuple

from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    translate_field_type,
)
from dspy.clients.base_lm import BaseLM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import LMError

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


def _chat_format():
    from dspy.adapters._engine.formats.chat import ChatFormat

    return ChatFormat()


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    """Default Adapter for most language models.

    The ChatAdapter formats DSPy signatures into a format compatible with most language models.
    It uses delimiter patterns like `[[ ## field_name ## ]]` to clearly separate input and output fields in
    the message content.

    Key features:
        - Structures inputs and outputs using field header markers for clear field delineation.
        - Provides automatic fallback to JSONAdapter if the chat format fails.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[type]] | None = None,
        use_json_adapter_fallback: bool = True,
        parallel_tool_calls: bool | None = None,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during adapter methods.
            use_native_function_calling: Whether to enable native function calling capabilities.
            native_response_types: List of output field types handled by native LM features.
            use_json_adapter_fallback: Whether to automatically fallback to JSONAdapter if the ChatAdapter fails.
                If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using
                JSONAdapter. Defaults to True.
            parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
                calling is active. If None, the adapter does not set the provider option.
        """
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = use_json_adapter_fallback

    def _make_json_adapter_fallback(self):
        from dspy.adapters.json_adapter import JSONAdapter

        return JSONAdapter(
            use_native_function_calling=self.use_native_function_calling,
            parallel_tool_calls=self.parallel_tool_calls,
        )

    def _should_reraise_instead_of_fallback(self, error: Exception) -> bool:
        """The fallback guard, implemented once for the sync and async paths.

        On LM errors, when already a JSONAdapter, or when the fallback is
        disabled, the ORIGINAL error propagates instead of retrying with a
        different adapter.
        """
        from dspy.adapters.json_adapter import JSONAdapter

        return isinstance(error, LMError) or isinstance(self, JSONAdapter) or not self.use_json_adapter_fallback

    def __call__(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if self._should_reraise_instead_of_fallback(e):
                raise
            return self._make_json_adapter_fallback()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if self._should_reraise_instead_of_fallback(e):
                raise
            return await self._make_json_adapter_fallback().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_description(self, signature: type[Signature]) -> str:
        return _chat_format().render_field_description(signature)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        `ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
        `[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
        output fields section to indicate the end of the output fields.

        Body kept (not delegated to ChatFormat): it dispatches through the
        overridable `format_field_with_value` hook, which override-routed
        subclasses customize.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def format_task_description(self, signature: type[Signature]) -> str:
        return _chat_format().render_task_description(signature)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        # Body kept (not delegated): per-block join semantics and the
        # overridable user_message_output_requirements hook must stay
        # byte-exact for override-routed subclasses.
        messages = [prefix]
        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Returns a simplified format reminder for the language model.

        In chat-based interactions, language models may lose track of the required output format
        as the conversation context grows longer. This method generates a concise reminder of
        the expected output structure that can be included in user messages.

        Args:
            signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

        Returns:
            str: A simplified description of the required output format.

        Note:
            This is a more lightweight version of `format_field_structure` specifically designed
            for inline reminders within chat messages.
        """

        return _chat_format().output_requirements(signature)

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        assistant_message_content = self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        from dspy.adapters._engine.overrides import resolve_override_verdict

        # Engine-backed classes parse via the resolved Format — the SAME
        # object that rendered the request, so format and parse share one
        # source of truth. Override-routed instances keep the legacy body.
        # The branch lives inside parse() so callback dispatch is identical.
        if resolve_override_verdict(self).engine_eligible:
            from dspy.adapters._engine.formats import resolve_format

            fmt = resolve_format(self)
            if fmt is not None:
                return fmt.parse(signature, completion)

        # Single source of truth: the legacy body is the same ChatFormat
        # parse the engine path resolves (a true leaf — no overridable hook
        # dispatch — so delegation cannot bypass subclass customizations).
        return _chat_format().parse(signature, completion)

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding
                value.

        Returns:
            The joined formatted values of the fields, represented as a string
        """
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """
        Format the call data into finetuning data according to the OpenAI API specifications.

        For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
        with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
        wrapped in a dictionary with a "messages" key.
        """
        system_user_messages = self.format(  # returns a list of dicts with the keys "role" and "content"
            signature=signature, demos=demos, inputs=inputs
        )
        assistant_message_content = self.format_assistant_message_content(  # returns a string, without the role
            signature=signature, outputs=outputs
        )
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
