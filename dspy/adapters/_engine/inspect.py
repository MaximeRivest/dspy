"""Internal plan pretty-printer.

Undocumented developer tooling: render a plan's contents as readable text
while debugging the engine. This is deliberately NOT public — a future epic
may grow it into ``adapter.explain()``/``preview()``, attached at the
render-only plan entry point.
"""

from dspy.adapters._engine.ir import AdapterPlan


def describe_plan(plan: AdapterPlan) -> str:
    lines = ["AdapterPlan"]

    def section(title, rows):
        if rows:
            lines.append(f"  {title}:")
            lines.extend(f"    {row}" for row in rows)

    section(
        "input_fields",
        [
            f"{f.name}"
            + (f" (from {f.original_name})" if f.original_name and f.original_name != f.name else "")
            + (" [hidden]" if f.hidden else "")
            for f in plan.input_fields
        ],
    )
    section(
        "output_fields",
        [
            f"{f.name}"
            + (f" -> {f.destination_name}" if f.destination_name != f.name else "")
            + (" [hidden]" if f.hidden else "")
            for f in plan.output_fields
        ],
    )
    section("field_transforms", [repr(t) for t in plan.field_transforms])
    section("tools", [str(t.get("function", {}).get("name", t)) if isinstance(t, dict) else str(t) for t in plan.tools])
    section("metadata", [f"{k}={v!r}" for k, v in plan.metadata.items()])
    section("parsers", [getattr(p, "name", repr(p)) for p in plan.parsers])
    section("warnings", plan.warnings)
    section(
        "debug_links",
        [f"{link.source} --{link.strategy}--> {link.destination}" for link in plan.debug_links],
    )
    if len(lines) == 1:
        lines.append("  (empty)")
    return "\n".join(lines)
