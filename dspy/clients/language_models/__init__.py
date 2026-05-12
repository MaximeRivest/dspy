"""Normalized language model implementations and types."""

from dspy.clients.language_models.base import LanguageModel, LMCapabilities
from dspy.clients.language_models.router import LM, LMRouter, register_lm_backend
from dspy.clients.language_models.features import FeatureStatus, LMFeatureReporter, LMRequestSupport, LMSupportIssue
from dspy.clients.language_models.types import (
    Assistant,
    AsyncLMStream,
    Developer,
    LMAudioDelta,
    LMAudioPart,
    LMBasePart,
    LMCacheConfig,
    LMCitationDelta,
    LMCitationPart,
    LMConfig,
    LMDelta,
    LMFilePart,
    LMHistoryEntry,
    LMImageDelta,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMOutputBuilder,
    LMPart,
    LMPromptCacheConfig,
    LMReasoningConfig,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMStream,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMTextPart,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
    System,
    ToolCall,
    ToolResult,
    User,
)


def __getattr__(name: str):
    if name in {"LiteLLMChatLM", "LiteLLMTextLM", "LiteLLMResponsesLM"}:
        from dspy.clients.language_models.litellm import LiteLLMChatLM, LiteLLMResponsesLM, LiteLLMTextLM

        return {
            "LiteLLMChatLM": LiteLLMChatLM,
            "LiteLLMTextLM": LiteLLMTextLM,
            "LiteLLMResponsesLM": LiteLLMResponsesLM,
        }[name]
    if name == "LM15LM":
        from dspy.clients.language_models.lm15 import LM15LM

        return LM15LM
    if name in {"OpenAIChatLM", "CompletionLM", "OpenAIResponsesLM", "ResponsesLM", "OpenAITextLM", "TextCompletionLM"}:
        from dspy.clients.language_models.openai_format import (
            CompletionLM,
            OpenAIChatLM,
            OpenAIResponsesLM,
            OpenAITextLM,
            ResponsesLM,
            TextCompletionLM,
        )

        return {
            "OpenAIChatLM": OpenAIChatLM,
            "CompletionLM": CompletionLM,
            "OpenAIResponsesLM": OpenAIResponsesLM,
            "ResponsesLM": ResponsesLM,
            "OpenAITextLM": OpenAITextLM,
            "TextCompletionLM": TextCompletionLM,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LanguageModel",
    "LMCapabilities",
    "LM",
    "LMRouter",
    "register_lm_backend",
    "FeatureStatus",
    "LMFeatureReporter",
    "LMRequestSupport",
    "LMSupportIssue",
    "LiteLLMChatLM",
    "LiteLLMTextLM",
    "LiteLLMResponsesLM",
    "LM15LM",
    "OpenAIChatLM",
    "CompletionLM",
    "OpenAIResponsesLM",
    "ResponsesLM",
    "OpenAITextLM",
    "TextCompletionLM",
    "LMBasePart",
    "LMTextPart",
    "LMImagePart",
    "LMAudioPart",
    "LMFilePart",
    "LMToolCallPart",
    "LMToolResultPart",
    "LMThinkingPart",
    "LMCitationPart",
    "LMRefusalPart",
    "LMPart",
    "LMMessage",
    "LMToolSpec",
    "LMReasoningConfig",
    "LMToolChoice",
    "LMCacheConfig",
    "LMPromptCacheConfig",
    "LMConfig",
    "LMRequest",
    "LMUsage",
    "LMOutput",
    "LMResponse",
    "LMDelta",
    "LMTextDelta",
    "LMThinkingDelta",
    "LMToolCallDelta",
    "LMCitationDelta",
    "LMImageDelta",
    "LMAudioDelta",
    "LMStreamEvent",
    "LMStreamStartEvent",
    "LMStreamDeltaEvent",
    "LMStreamOutputEndEvent",
    "LMStreamEndEvent",
    "LMStreamErrorEvent",
    "LMOutputBuilder",
    "LMStream",
    "AsyncLMStream",
    "LMHistoryEntry",
    "System",
    "Developer",
    "User",
    "Assistant",
    "ToolCall",
    "ToolResult",
]
