"""Declared capacity: what a template can host textually, derived statically.

The template language is closed precisely so this analysis is possible
(spec section 2): bake's triple check — signature roles x LM capabilities x
template capacity — needs to know, without rendering, whether a textual
strategy has anywhere to land. Nothing consumes this yet; the strategy-
awareness PR (D-4) does.
"""

from dataclasses import dataclass

from dspy.adapters._engine.template.parser import (
    AggregateSlot,
    ContentMessage,
    DemosDirective,
    FragmentsSlot,
    HistoryDirective,
    Loop,
    ParsedTemplate,
    Section,
    ValueSlot,
)


@dataclass(frozen=True)
class TemplateCapacity:
    """A template's textual hosting surface, as data.

    ``iterates_outputs`` is the load-bearing bit for output-side textual
    strategies (a reasoning/citations polyfill lands as an extra visible
    field, which only renders if the template iterates outputs);
    ``iterates_inputs`` plays the same part for input-side textual
    rendering; ``fragment_targets`` names the slots textual strategies can
    fill with instructions.
    """

    iterates_inputs: bool
    iterates_outputs: bool
    fragment_targets: frozenset[str]
    hosts_demos: bool
    hosts_history: bool
    field_slots: frozenset[str]

    def hosts_role_textually(self, role: str) -> bool:
        """Can a textual strategy for ``role`` land in this template?"""
        from dspy.signatures.field import SEMANTIC_ROLES

        if role not in SEMANTIC_ROLES:
            raise ValueError(f"unknown semantic role {role!r} — valid roles: {', '.join(SEMANTIC_ROLES)}")
        if role == "history":
            return self.hosts_history
        if role in ("media", "tools"):
            return self.iterates_inputs or bool(self.field_slots)
        # Output-side roles (reasoning, citations, tool_calls, code, plain)
        # polyfill as visible output fields.
        return self.iterates_outputs


def declared_capacity(template: ParsedTemplate) -> TemplateCapacity:
    """Analyze a parsed template's slots into its textual capacity."""
    iterates_inputs = False
    iterates_outputs = False
    fragment_targets: set[str] = set()
    hosts_demos = False
    hosts_history = False
    field_slots: set[str] = set()

    def visit(nodes) -> None:
        nonlocal iterates_inputs, iterates_outputs, hosts_demos, hosts_history
        for node in nodes:
            if isinstance(node, Loop):
                if node.collection == "inputs":
                    iterates_inputs = True
                else:
                    iterates_outputs = True
                visit(node.body)
            elif isinstance(node, Section):
                visit(node.body)
            elif isinstance(node, AggregateSlot):
                if node.kind == "inputs":
                    iterates_inputs = True
                elif node.kind == "outputs":
                    iterates_outputs = True
                elif node.kind == "demos":
                    hosts_demos = True
                else:
                    hosts_history = True
            elif isinstance(node, FragmentsSlot):
                fragment_targets.add(node.target)
            elif isinstance(node, ValueSlot):
                field_slots.add(node.name)

    for message in template.messages:
        if isinstance(message, ContentMessage):
            visit(message.nodes)
        elif isinstance(message, DemosDirective):
            hosts_demos = True
        elif isinstance(message, HistoryDirective):
            hosts_history = True

    return TemplateCapacity(
        iterates_inputs=iterates_inputs,
        iterates_outputs=iterates_outputs,
        fragment_targets=frozenset(fragment_targets),
        hosts_demos=hosts_demos,
        hosts_history=hosts_history,
        field_slots=frozenset(field_slots),
    )
