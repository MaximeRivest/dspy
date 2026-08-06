"""The constrained template language (adapter IR contract, section 3).

Engine-private, extraction-ready. A template is a list of messages whose
content strings use a closed vocabulary — value slots, aggregate slots,
loop blocks, positional fragment slots, escapes — plus ``demos``/``history``
directive messages. The whole vocabulary is one data structure
(``vocabulary.VOCABULARY``); the parser validates eagerly with teaching
errors; rendering is pure; capacity derives statically; ``preview()``
renders full message lists with no LM call.
"""

from dspy.adapters._engine.template.capacity import TemplateCapacity, declared_capacity
from dspy.adapters._engine.template.parser import (
    ContentMessage,
    DemosDirective,
    HistoryDirective,
    ParsedTemplate,
    TemplateError,
    parse_content,
    parse_message_template,
)
from dspy.adapters._engine.template.preview import preview, render_template_messages
from dspy.adapters._engine.template.renderer import (
    RenderContext,
    TemplateRenderError,
    classify_demos,
    render_nodes,
    render_user_content,
)
from dspy.adapters._engine.template.vocabulary import VOCABULARY, describe_template_language

__all__ = [
    "VOCABULARY",
    "ContentMessage",
    "DemosDirective",
    "HistoryDirective",
    "ParsedTemplate",
    "RenderContext",
    "TemplateCapacity",
    "TemplateError",
    "TemplateRenderError",
    "classify_demos",
    "declared_capacity",
    "describe_template_language",
    "parse_content",
    "parse_message_template",
    "preview",
    "render_nodes",
    "render_template_messages",
    "render_user_content",
]
