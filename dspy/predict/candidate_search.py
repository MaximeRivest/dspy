"""Shared candidate-search engine behind ``dspy.BestOfN`` and ``dspy.Refine``.

This module separates *candidate search* from *module execution* and *adapter
transport*:

- The **search plane** (this module) decides which candidates to run. It clones
  the wrapped program per attempt, diversifies sampling through LM rollout ids,
  scores predictions with a user reward function, applies an explicit stopping
  rule and failure budget, and optionally realizes non-authoritative ``Hint``s
  proposed between attempts into the *next* candidate.
- The **execution plane** is the candidate program itself: an ordinary
  ``dspy.Module`` clone whose calls are fully defined (signature + inputs)
  before any transport runs.
- The **transport plane** (Adapters) only formats and parses already-defined
  calls. The search plane never wraps, subclasses, or reconfigures adapters: a
  hint becomes an ordinary defaulted input field on the candidate's own
  signature, so adapter resolution (call > instance > settings > default) is
  untouched and any future adapter implementation works unchanged.

Hints are fallible search hypotheses, not authoritative intent: they carry
provenance, may be ignored, and are only realized on throwaway candidate
clones. Hintable call sites are a *runtime* notion discovered per candidate; a
module may expose zero sites (e.g. a leaf that holds its implementation as
opaque state rather than as ``dspy.Predict`` sub-modules), in which case the
search degrades to resample-and-select and proposed hints are recorded as
unapplied. The engine therefore does not assume that every runtime predictor is
an optimizer ``Parameter``, nor that every refinable leaf is a ``dspy.Predict``.

Every search produces a ``SearchRecord`` describing each attempt (rollout id,
realized hints, reward, error, raw trace), which the public facades attach to
the returned prediction for inspection via ``search_record(prediction)``. The
caller-visible trace, by contrast, contains only the *selected* execution,
expressed in terms of the original program: hint fields injected by the search
are stripped and trace entries are remapped to the original predictors.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures import InputField

logger = logging.getLogger(__name__)

# Name of the transient input field a realized hint occupies on a candidate's
# signature. Kept as `hint_` for continuity with earlier Refine releases.
HINT_FIELD_NAME = "hint_"

_HINT_FIELD_DESC = "A hint to the module from an earlier run"


@dataclass
class Hint:
    """A fallible, non-authoritative suggestion produced during candidate search.

    A hint nudges the next candidate; it is not user intent and carries no
    authority. The search plane realizes it into an *effective candidate*
    (a throwaway clone whose targeted call site gains a defaulted
    ``hint_`` input field) before any Adapter sees the call. A hint whose
    target cannot be resolved on the candidate is kept for provenance with
    ``applied=False`` rather than being forced through transport.

    Attributes:
        target: Name of the call site inside the candidate program (as reported
            by the candidate's predictor discovery) this hint addresses.
        content: The natural-language suggestion.
        authority: Always ``"non_authoritative"``; recorded explicitly so the
            provenance of injected text is never ambiguous.
        scope: Lifetime of the hint. ``"next_candidate"`` means it is realized
            into the immediately following candidate only.
        provenance: Where the hint came from (e.g. proposer name, the attempt
            index and reward that motivated it).
        applied: Set by the engine: whether the hint was realized into the
            candidate it was proposed for.
    """

    target: str
    content: str
    authority: str = "non_authoritative"
    scope: str = "next_candidate"
    provenance: dict[str, Any] = field(default_factory=dict)
    applied: bool = False


@dataclass
class Attempt:
    """One candidate execution: its identity, realized hints, and outcome.

    ``trace`` is refinement metadata: the raw trace of the candidate clone,
    including any ``hint_`` inputs. The caller-visible program trace is derived
    separately (see ``CandidateSearch.export_trace``).
    """

    index: int
    rollout_id: Any
    hints: list[Hint] = field(default_factory=list)
    prediction: Prediction | None = None
    reward: float | None = None
    trace: list = field(default_factory=list)
    error: Exception | None = None
    proposal_error: Exception | None = None
    # id(predictor-in-candidate) -> dotted name; used to remap the selected
    # trace back onto the original program's predictors.
    predictor_names: dict[int, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.prediction is not None


@dataclass
class SearchRecord:
    """Full provenance of one search: every attempt plus the selected index."""

    attempts: list[Attempt] = field(default_factory=list)
    best_index: int | None = None

    @property
    def best(self) -> Attempt | None:
        return self.attempts[self.best_index] if self.best_index is not None else None

    @property
    def best_reward(self) -> float | None:
        best = self.best
        return best.reward if best is not None else None


def search_record(prediction: Prediction) -> SearchRecord | None:
    """Return the ``SearchRecord`` attached to a prediction by ``BestOfN``/``Refine``, if any."""
    return getattr(prediction, "_search_record", None)


class CandidateSearch:
    """Sequential candidate search over clones of a wrapped module.

    The engine embodies exactly two policies through one optional hook:
    independent sampling (``propose=None``, used by ``BestOfN``) and sequential
    hint-guided refinement (``propose`` returns ``Hint``s consumed by the next
    candidate, used by ``Refine``). Attempts are strictly sequential so that a
    hint proposed after attempt ``i`` can inform attempt ``i + 1``.

    Guarantees:

    - The original module is never mutated: every attempt runs on a deep copy
      with an LM copy at ``rollout_id = start + i, temperature=1.0``.
    - Adapters are never wrapped or replaced; hints are realized as defaulted
      input fields on the candidate's own signatures, so every call an Adapter
      receives is already fully defined.
    - Auxiliary work (reward scoring, hint proposal) runs in an isolated trace
      context and never reaches the caller's trace.
    - The failure budget (``fail_count``) is local to one ``run``/``arun`` call;
      the engine raises the last error if it is exceeded, or if the search ends
      with no successful attempt at all.

    Args:
        module: The module whose candidates are searched. It only needs to be a
            callable ``dspy.Module``; it may expose zero ``dspy.Predict``
            leaves, in which case hints degrade to provenance-only.
        num_candidates: Maximum number of attempts.
        reward_fn: ``(inputs_dict, prediction) -> float`` scoring function.
        threshold: Stop as soon as an attempt's reward reaches this value.
        fail_count: Number of failed attempts tolerated per search before the
            last error is raised. Defaults to ``num_candidates``.
        propose: Optional hook ``(attempt, inputs, candidate) -> Sequence[Hint]``
            invoked after each scored-but-below-threshold attempt (except the
            last). Exceptions from the hook are recorded on the attempt and do
            not consume the failure budget: a hint is a fallible aid, and its
            failure must not abort the search.
    """

    def __init__(
        self,
        module: Module,
        num_candidates: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float | None,
        fail_count: int | None = None,
        propose: Callable[[Attempt, dict, Module], Sequence[Hint]] | None = None,
    ):
        self.module = module
        self.num_candidates = num_candidates
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.fail_count = fail_count if fail_count is not None else num_candidates
        self.propose = propose

    # -- search ------------------------------------------------------------

    def run(self, inputs: dict[str, Any]) -> SearchRecord:
        """Run the search synchronously and return its full record."""
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        record = SearchRecord()
        failures, last_error = 0, None
        pending: list[Hint] = []

        for idx in range(self.num_candidates):
            candidate, attempt = self._prepare_candidate(lm, idx, start + idx, pending)
            pending = []
            record.attempts.append(attempt)
            try:
                with settings.context(trace=[]):
                    try:
                        prediction = candidate(**inputs)
                    finally:
                        attempt.trace = settings.trace.copy()
                    # Scored after the trace copy so judge/reward LM calls stay
                    # out of the candidate's trace (and the exported one).
                    reward = self.reward_fn(inputs, prediction)
            except Exception as e:
                failures, last_error = failures + 1, e
                attempt.error = e
                logger.warning("CandidateSearch: attempt %d (rollout id %s) failed: %s", idx + 1, attempt.rollout_id, e)
                if failures > self.fail_count:
                    raise
                continue

            if self._record_scored(record, attempt, prediction, reward):
                break
            if idx < self.num_candidates - 1:
                pending = self._propose_hints(attempt, inputs, candidate)

        if record.best_index is None and last_error is not None:
            raise last_error
        return record

    async def arun(self, inputs: dict[str, Any]) -> SearchRecord:
        """Async variant of ``run``; the wrapped module is invoked via ``acall``.

        The ``reward_fn`` and ``propose`` hooks remain synchronous callables.
        """
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        record = SearchRecord()
        failures, last_error = 0, None
        pending: list[Hint] = []

        for idx in range(self.num_candidates):
            candidate, attempt = self._prepare_candidate(lm, idx, start + idx, pending)
            pending = []
            record.attempts.append(attempt)
            try:
                with settings.context(trace=[]):
                    try:
                        prediction = await candidate.acall(**inputs)
                    finally:
                        attempt.trace = settings.trace.copy()
                    reward = self.reward_fn(inputs, prediction)
            except Exception as e:
                failures, last_error = failures + 1, e
                attempt.error = e
                logger.warning("CandidateSearch: attempt %d (rollout id %s) failed: %s", idx + 1, attempt.rollout_id, e)
                if failures > self.fail_count:
                    raise
                continue

            if self._record_scored(record, attempt, prediction, reward):
                break
            if idx < self.num_candidates - 1:
                pending = self._propose_hints(attempt, inputs, candidate)

        if record.best_index is None and last_error is not None:
            raise last_error
        return record

    def finalize(self, record: SearchRecord) -> Prediction | None:
        """Publish the selected attempt: extend the caller's trace and attach provenance.

        The caller-visible trace receives only the selected execution, remapped
        onto the original program (see ``export_trace``). The complete search
        record, including failed attempts and raw hinted traces, is attached to
        the returned prediction and retrievable with ``search_record(pred)``.
        """
        best = record.best
        if best is None:
            return None
        if best.trace and settings.trace is not None:
            settings.trace.extend(self.export_trace(best))
        prediction = best.prediction
        prediction._search_record = record
        return prediction

    def export_trace(self, attempt: Attempt) -> list:
        """The selected attempt's trace, expressed in terms of the original program.

        Trace entries are remapped by name from the candidate clone's
        predictors to the original module's predictors, and the transient
        ``hint_`` input injected by the search is stripped (unless the original
        signature genuinely declares such a field). The unmodified trace stays
        available on the ``Attempt`` as refinement metadata.
        """
        original_by_name = dict(self.module.named_predictors())
        exported = []
        for predictor, trace_inputs, outputs in attempt.trace:
            name = attempt.predictor_names.get(id(predictor))
            target = original_by_name.get(name, predictor)
            if (
                target is not predictor
                and HINT_FIELD_NAME in trace_inputs
                and HINT_FIELD_NAME not in target.signature.input_fields
            ):
                trace_inputs = {k: v for k, v in trace_inputs.items() if k != HINT_FIELD_NAME}
            exported.append((target, trace_inputs, outputs))
        return exported

    # -- internals ---------------------------------------------------------

    def _resolve_lm(self):
        # `get_lm()` raises for zero predictors as well as for mixed LMs; only
        # the latter is a real ambiguity, so consult it only when there are
        # predictors to consult. A module with no Predict leaves still searches
        # fine with the ambient LM driving rollout diversity for any LM calls
        # it makes through settings.
        lm = (self.module.get_lm() if self.module.predictors() else None) or settings.lm
        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`, or set one on "
                "the wrapped module with `module.set_lm(...)`."
            )
        return lm

    def _prepare_candidate(self, lm, index: int, rollout_id, hints: list[Hint]) -> tuple[Module, Attempt]:
        candidate = self.module.deepcopy()
        candidate.set_lm(lm.copy(rollout_id=rollout_id, temperature=1.0))
        realized = [self._realize_hint(candidate, hint) for hint in hints]
        attempt = Attempt(
            index=index,
            rollout_id=rollout_id,
            hints=realized,
            predictor_names={id(p): name for name, p in candidate.named_predictors()},
        )
        return candidate, attempt

    def _realize_hint(self, candidate: Module, hint: Hint) -> Hint:
        """Realize one hint into the candidate program, before any Adapter runs.

        The targeted call site's signature gains a ``hint_`` input field whose
        default carries the hint content, so `Predict` fills it like any other
        defaulted input and the Adapter receives an already-defined call. Only
        the throwaway candidate clone is modified. Hintable sites are currently
        the candidate's ``dspy.Predict`` leaves; this is a per-candidate runtime
        discovery, not a claim that predictors are optimizer parameters, and a
        target that does not resolve leaves the hint recorded as unapplied.
        """
        site = self._hintable_sites(candidate).get(hint.target)
        if site is None:
            hint.applied = False
            logger.debug("CandidateSearch: hint target %r not found on candidate; keeping hint unapplied.", hint.target)
            return hint
        if HINT_FIELD_NAME in site.signature.input_fields:
            # The program genuinely declares a field with the reserved name;
            # refuse to shadow declared intent with a search hint.
            hint.applied = False
            logger.warning(
                "CandidateSearch: predictor %r already declares an input named %r; hint left unapplied.",
                hint.target,
                HINT_FIELD_NAME,
            )
            return hint
        site.signature = site.signature.append(
            HINT_FIELD_NAME, InputField(default=hint.content, desc=_HINT_FIELD_DESC), type_=str
        )
        hint.applied = True
        return hint

    def _hintable_sites(self, candidate: Module) -> dict[str, Predict]:
        """Call sites of this candidate that can absorb a call-scoped hint.

        Today these are the candidate's ``dspy.Predict`` leaves, because Predict
        is the leaf whose calls are defined by a Signature the search can extend.
        Leaves that execute a Signature through other means (e.g. code state run
        in an interpreter) legitimately expose zero sites; the search must not
        manufacture one through transport.
        """
        return dict(candidate.named_predictors())

    def _record_scored(self, record: SearchRecord, attempt: Attempt, prediction: Prediction, reward: float) -> bool:
        attempt.prediction = prediction
        attempt.reward = reward
        if record.best_index is None or reward > record.attempts[record.best_index].reward:
            record.best_index = attempt.index
        return self.threshold is not None and reward >= self.threshold

    def _propose_hints(self, attempt: Attempt, inputs: dict[str, Any], candidate: Module) -> list[Hint]:
        if self.propose is None:
            return []
        try:
            # Proposal is auxiliary search work; keep its LM calls (if any) out
            # of the caller's trace.
            with settings.context(trace=[]):
                return list(self.propose(attempt, inputs, candidate) or [])
        except Exception as e:
            attempt.proposal_error = e
            logger.warning("CandidateSearch: hint proposal after attempt %d failed: %s", attempt.index + 1, e)
            return []
