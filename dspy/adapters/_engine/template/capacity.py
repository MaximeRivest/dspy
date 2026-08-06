"""Declared capacity: what a template can host textually, derived statically.

The template language is closed precisely so this analysis is possible
(spec section 2): bake's triple check — signature roles x LM capabilities x
template capacity — needs to know, without rendering, whether a textual
strategy has anywhere to land. Nothing consumes this yet; the strategy-
awareness PR (D-4) does.

Two lanes, kept apart on purpose:

- the **live-call lane** (content messages) is where the current call's
  inputs render and the response format is taught — ``iterates_inputs``,
  ``iterates_outputs``, and ``field_slots`` describe it;
- the **example lane** (demos/history directive patterns, including the
  language's default patterns for bare directives) renders only when
  demos or turns exist at call time — ``directive_iterates_inputs``,
  ``directive_iterates_outputs``, and ``directive_field_slots`` describe
  it, and it never counts toward live-lane hosting: a demo pattern cannot
  place a live call's field.
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
    directive_pair,
)


@dataclass(frozen=True)
class TemplateCapacity:
    """A template's textual hosting surface, as data.

    ``iterates_outputs`` is the load-bearing bit for output-side textual
    strategies (a reasoning/citations polyfill lands as an extra visible
    field, which only renders if the template iterates outputs);
    ``iterates_inputs`` plays the same part for input-side textual
    rendering; ``fragment_targets`` names the slots textual strategies can
    fill with instructions. The ``directive_*`` fields mirror the first
    three for the example lane (demos/history patterns) — the substrate
    for bake-time checks on hidden fields referenced from demo patterns
    (ADP-007), never a substitute for live-lane hosting.
    """

    iterates_inputs: bool
    iterates_outputs: bool
    fragment_targets: frozenset[str]
    hosts_demos: bool
    hosts_history: bool
    field_slots: frozenset[str]
    directive_iterates_inputs: bool = False
    directive_iterates_outputs: bool = False
    directive_field_slots: frozenset[str] = frozenset()

    def hosts_role_textually(self, role: str, field: str | None = None) -> bool:
        """Can a textual strategy for ``role`` land in this template?

        ``media`` and ``tools`` are per-field questions — an input field
        lands textually only where an inputs iteration or its own named
        slot places it, so pass the field's name. Asking without one
        refuses: the coarse per-role answer over-claimed (any unrelated
        slot counted), which is exactly the silent degradation ADP-006's
        bake check exists to refuse.
        """
        from dspy.signatures.field import SEMANTIC_ROLES

        if role not in SEMANTIC_ROLES:
            raise ValueError(f"unknown semantic role {role!r} — valid roles: {', '.join(SEMANTIC_ROLES)}")
        if role == "history":
            return self.hosts_history
        if role in ("media", "tools"):
            if field is None:
                raise ValueError(
                    f"{role} capacity is a per-field question — call "
                    f"hosts_role_textually({role!r}, field='name'): a field lands textually "
                    "only where an inputs iteration or its own named slot places it"
                )
            return self.iterates_inputs or field in self.field_slots
        # Output-side roles (reasoning, citations, tool_calls, code, plain)
        # polyfill as visible output fields.
        return self.iterates_outputs


def declared_capacity(template: ParsedTemplate) -> TemplateCapacity:
    """Analyze a parsed template's slots into its textual capacity.

    Directive patterns are analyzed through the same resolution rendering
    uses (own pair, else the demos directive's, else the language default
    patterns), so a bare directive's capacity reflects what it actually
    renders.
    """
    live = {"iterates_inputs": False, "iterates_outputs": False}
    example = {"iterates_inputs": False, "iterates_outputs": False}
    live_slots: set[str] = set()
    example_slots: set[str] = set()
    fragment_targets: set[str] = set()
    hosts_demos = False
    hosts_history = False

    def visit(nodes, lane: dict, slots: set) -> None:
        nonlocal hosts_demos, hosts_history
        for node in nodes:
            if isinstance(node, Loop):
                lane["iterates_" + node.collection] = True
                visit(node.body, lane, slots)
            elif isinstance(node, Section):
                visit(node.body, lane, slots)
            elif isinstance(node, AggregateSlot):
                if node.kind in ("inputs", "outputs"):
                    lane["iterates_" + node.kind] = True
                elif node.kind == "demos":
                    hosts_demos = True
                else:
                    hosts_history = True
            elif isinstance(node, FragmentsSlot):
                fragment_targets.add(node.target)
            elif isinstance(node, ValueSlot):
                slots.add(node.name)

    demos_directive = template.demos_directive
    for message in template.messages:
        if isinstance(message, ContentMessage):
            visit(message.nodes, live, live_slots)
        elif isinstance(message, (DemosDirective, HistoryDirective)):
            if isinstance(message, DemosDirective):
                hosts_demos = True
            else:
                hosts_history = True
            user_nodes, assistant_nodes = directive_pair(message, demos_directive)
            visit(user_nodes, example, example_slots)
            visit(assistant_nodes, example, example_slots)

    return TemplateCapacity(
        iterates_inputs=live["iterates_inputs"],
        iterates_outputs=live["iterates_outputs"],
        fragment_targets=frozenset(fragment_targets),
        hosts_demos=hosts_demos,
        hosts_history=hosts_history,
        field_slots=frozenset(live_slots),
        directive_iterates_inputs=example["iterates_inputs"],
        directive_iterates_outputs=example["iterates_outputs"],
        directive_field_slots=frozenset(example_slots),
    )
