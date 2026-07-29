"""Parameter and refinement-unit protocol.

DSPy programs live in three distinct graphs that should not be conflated:

- The **module graph** describes composition: which modules contain which
  sub-modules. It is what ``named_sub_modules`` walks.
- The **parameter graph** describes optimizer state: the atomic units of state
  an optimizer may read and update. A :class:`Parameter` is a node in this
  graph. It is what ``named_parameters`` walks.
- The **runtime graph** describes execution: the objects that happen to run
  when a program is called. Runtime objects may be *derived* from a
  parameter's state (for example, predictors that a code-holding leaf
  constructs from its own source) without themselves being parameters.

A ``dspy.Predict`` is a leaf that belongs to all three graphs at once, which
historically encouraged the assumption ``Predict == leaf == Parameter ==
refinable state``. That identity is accidental, not essential: a leaf may hold
optimizable state (e.g. a source string executed in a sandbox) while exposing
zero independently-updatable predictors. The protocol below lets refinement
target *declared* flexible state instead of traversing runtime objects.

Refinement-unit protocol
------------------------

A refinement unit is a :class:`Parameter` that opts in to receiving
non-authoritative, call-time hints (see :class:`Hint`). The default is to
decline: committed state must not silently become a feedback channel. Search
components such as ``dspy.Refine`` discover units through
``Module.named_refinement_units()``, which treats every :class:`Parameter` as
an atomic boundary — state reachable *through* a unit is derived runtime state
owned by that unit, never an independent refinement target.

A hint is interpreted by the unit itself into an effective candidate *before*
execution-plane components (such as Adapters) see the call. Adapters render
and parse already-defined calls; they do not invent semantic fields.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Hint:
    """A fallible, non-authoritative suggestion addressed to one refinement unit.

    A hint is search-plane data, not program intent: it may help or mislead,
    and a unit is always free to ignore it. Hints therefore carry explicit
    provenance and never gain call-time authority by being smuggled into
    transport (e.g. by mutating an Adapter).

    Args:
        target: Name of the refinement unit this hint addresses, as reported by
            ``Module.named_refinement_units()``.
        content: The advice text.
        authority: Epistemic status of the hint. DSPy currently defines only
            ``"non_authoritative"``; units must treat any hint as ignorable.
        scope: Intended lifetime. ``"next_candidate"`` means the hint applies to
            the candidate (typically a throwaway deep copy) it is installed on
            and must never persist into saved state or the original program.
        provenance: Where the hint came from (e.g. the critic that produced it,
            the reward and rollout that prompted it).
    """

    target: str
    content: str
    authority: str = "non_authoritative"
    scope: str = "next_candidate"
    provenance: Mapping[str, Any] | None = field(default=None)


class Parameter:
    """Marker base class for a leaf whose state is an optimizer's atomic update unit.

    Being a ``Parameter`` says: "an optimizer may treat this object's declared
    state as one unit — read it, replace it, serialize it." It does *not* say
    that every runtime object the leaf creates is independently optimizable.
    Traversals that look for refinement targets stop at a ``Parameter``
    boundary.

    Subclasses opt in to call-time refinement by overriding
    :meth:`accepts_hint` and :meth:`apply_hint`. The defaults decline, so
    plain parameters (e.g. retrievers, code-holding leaves that have not
    declared a hint channel) are never silently injected with feedback.

    Note that traversal helpers such as ``named_parameters`` (and therefore
    ``set_lm``) also stop at a ``Parameter`` boundary: a leaf that constructs
    derived runtime state from its own state is responsible for propagating
    runtime configuration (LMs, interpreters, ...) into that state itself.
    """

    def accepts_hint(self, hint: "Hint | None" = None) -> bool:
        """Whether this unit accepts non-authoritative refinement hints.

        Args:
            hint: Optionally, the specific hint under consideration; units may
                accept hints in general but decline a particular one.

        Returns:
            ``False`` unless the subclass explicitly opts in.
        """
        return False

    def apply_hint(self, hint: "Hint") -> None:
        """Install a non-authoritative hint on this unit.

        Only called on units whose :meth:`accepts_hint` returned ``True``, and
        only on candidate copies whose lifetime matches the hint's ``scope``.
        The unit decides how (and whether) the hint influences its next
        execution; it must not alter the unit's persistent, serialized state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not accept refinement hints. "
            "Override `accepts_hint`/`apply_hint` to declare a hint channel."
        )

    def clear_hints(self) -> None:
        """Remove any installed hints. Safe to call on any unit."""

    def describe_refinement_unit(self) -> str:
        """A human/critic-readable description of this unit's I/O and intent."""
        return f"{self.__class__.__name__} (opaque refinement unit)"
