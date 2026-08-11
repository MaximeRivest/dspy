"""Stage A5 tests: optimizers as IR mutations with engine replay.

The loop shape under test: propose = mutate the program's data (demos,
instructions), score = replay the program through the engine over a
devset, keep = the best state — and checkpoint == save, so accepted
candidates are ordinary loadable artifacts. Scripts are callable
DummyLMs (answer tables keyed off the rendered prompt), so pass/fail is
deterministic whatever the call order.
"""

import json

import pytest

import dspy
from dspy import optim
from dspy.lm import BINDINGS


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


def chat_completion(**fields: str) -> str:
    """Render one scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


CAPITALS = {"France": "Paris", "Canada": "Ottawa", "Japan": "Tokyo", "Peru": "Lima"}


def make_example(country: str, answer: str | None = None) -> dspy.Example:
    return dspy.Example(
        question=f"What is the capital of {country}?",
        answer=answer if answer is not None else CAPITALS[country],
    ).with_inputs("question")


def oracle(table: dict[str, str]):
    """A CoT teacher: reads the country off the current turn, answers from the table."""

    def script(messages):
        current = str(messages[-1]["content"])
        country = next((c for c in CAPITALS if f"capital of {c}?" in current), None)
        return chat_completion(reasoning=f"Recall {country}.", answer=table.get(country, "unknown"))

    return script


def copycat(messages):
    """Answers correctly only when a demo in the prompt carries the capital."""
    context = "\n".join(str(m["content"]) for m in messages[:-1])
    current = str(messages[-1]["content"])
    country = next((c for c in CAPITALS if f"capital of {c}?" in current), None)
    answer = CAPITALS[country] if country and CAPITALS[country] in context else "unknown"
    return chat_completion(reasoning="Copy from the demos.", answer=answer)


def exact_answer(example, prediction):
    return example.answer == prediction.answer


def trainset_of(*countries: str) -> list[dspy.Example]:
    return [make_example(country) for country in countries]


# ---------------------------------------------------------------------------
# evaluate: engine replay over a devset
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_scores_and_counts_lm_calls(self):
        table = dict(CAPITALS)
        table.pop("Peru")  # the oracle answers "unknown" for Peru
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(table)))
        devset = trainset_of("France", "Japan", "Canada", "Peru")

        result = optim.evaluate(cot, devset, exact_answer)

        assert result.score == pytest.approx(0.75)
        assert len(result.results) == 4
        assert [score for _, _, score in result.results] == [1.0, 1.0, 1.0, 0.0]
        assert result.lm_calls == 4  # one engine predictor call per example

    def test_evaluate_failed_run_scores_zero(self):
        cot = dspy.ChainOfThought("question -> answer")
        script = [chat_completion(reasoning="ok", answer="Paris"), "no labeled fields at all"]
        dspy.configure(lm=dspy.DummyLM(script))
        devset = trainset_of("France", "Canada")

        result = optim.evaluate(cot, devset, exact_answer)

        assert result.score == pytest.approx(0.5)
        assert result.results[0][2] == 1.0
        assert result.results[1][1] is None  # the parse failure yields no prediction
        assert result.lm_calls == 1  # failed runs record no exhaust to count

    def test_evaluate_refusals_are_loud(self):
        cot = dspy.ChainOfThought("question -> answer")
        with pytest.raises(ValueError, match="non-empty devset"):
            optim.evaluate(cot, [], exact_answer)
        with pytest.raises(ValueError, match="with_inputs"):
            optim.evaluate(cot, [dspy.Example(question="q", answer="a")], exact_answer)

    def test_metric_exception_scores_that_example_zero_and_continues(self):
        # Brief 1.f: a metric that raises on ONE example scores that example
        # 0.0 and the devset pass continues — one flaky metric call must not
        # abort the whole run. The prediction ran fine, so its LM call still
        # counts.
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(dict(CAPITALS))))
        devset = trainset_of("France", "Japan", "Canada")

        def flaky_metric(example, prediction):
            if "Japan" in example.question:
                raise ValueError("metric exploded on Japan")
            return example.answer == prediction.answer

        with pytest.warns(UserWarning, match="metric raised on one example"):
            result = optim.evaluate(cot, devset, flaky_metric)

        # The run COMPLETED: France 1.0, Japan 0.0 (the raise), Canada 1.0.
        assert [score for _, _, score in result.results] == [1.0, 0.0, 1.0]
        assert result.score == pytest.approx(2 / 3)
        # The exploding example still has its prediction (only the metric
        # failed) and its engine call is counted.
        assert result.results[1][1] is not None
        assert result.lm_calls == 3


# ---------------------------------------------------------------------------
# LabeledFewShot: the trivial mutation proves the plumbing
# ---------------------------------------------------------------------------


class TestLabeledFewShot:
    def test_labeled_attaches_k_demos_in_ir_and_messages(self):
        trainset = trainset_of("France", "Japan", "Canada", "Peru")
        lm = dspy.DummyLM(oracle(CAPITALS))
        cot = dspy.ChainOfThought("question -> answer", lm=lm)

        compiled = optim.LabeledFewShot(k=2, seed=0).compile(cot, trainset=trainset)

        assert compiled is cot
        assert len(cot.demos) == 2

        # The demos are in the compiled IR, path-keyed with input designation.
        records = cot.to_manifest()["components"]["3b_demos"]["self"]
        assert records == [{**demo.toDict(), "input_keys": ["question"]} for demo in cot.demos]

        # And they change the rendered bytes: preview is the pure proof.
        adapter = cot.resolve_adapter()
        demo_dicts = [demo.toDict() for demo in cot.demos]
        preview = adapter.preview(
            cot.signature,
            inputs={"question": "What is the capital of France?"},
            demos=demo_dicts,
            capabilities=lm.capabilities,
        )
        rendered = "\n".join(str(message["content"]) for message in preview)
        for demo in demo_dicts:
            assert demo["question"] in rendered
            assert demo["answer"] in rendered

        # parse_preview reads back the exact convention the demos teach.
        fields = adapter.parse_preview(
            cot.signature,
            chat_completion(reasoning="Recall France.", answer="Paris"),
            capabilities=lm.capabilities,
        )
        assert fields == {"reasoning": "Recall France.", "answer": "Paris"}

        # The live call renders the same demos into the real exchange.
        prediction = cot(question="What is the capital of France?")
        assert prediction.answer == "Paris"
        live = "\n".join(str(message["content"]) for message in lm.calls[-1]["messages"])
        assert demo_dicts[0]["answer"] in live

    def test_labeled_is_deterministic_under_seed(self):
        trainset = trainset_of("France", "Japan", "Canada", "Peru")
        first = dspy.ChainOfThought("question -> answer")
        second = dspy.ChainOfThought("question -> answer")
        third = dspy.ChainOfThought("question -> answer")

        optim.LabeledFewShot(k=2, seed=0).compile(first, trainset=trainset)
        optim.LabeledFewShot(k=2, seed=0).compile(second, trainset=trainset)
        optim.LabeledFewShot(k=2, seed=1).compile(third, trainset=trainset)

        assert [d.toDict() for d in first.demos] == [d.toDict() for d in second.demos]
        assert [d.toDict() for d in first.demos] != [d.toDict() for d in third.demos]

    def test_labeled_unsampled_takes_the_first_k(self):
        trainset = trainset_of("France", "Japan", "Canada")
        cot = dspy.ChainOfThought("question -> answer")

        optim.LabeledFewShot(k=2).compile(cot, trainset=trainset, sample=False)

        assert [d.toDict() for d in cot.demos] == [e.toDict() for e in trainset[:2]]


# ---------------------------------------------------------------------------
# BootstrapFewShot: demos from metric-passing engine traces
# ---------------------------------------------------------------------------


class TestBootstrapFewShot:
    def test_bootstrap_demos_pass_the_metric(self):
        # The teacher answers Japan WRONG: that trace must not become a demo.
        table = {**CAPITALS, "Japan": "Osaka"}
        trainset = trainset_of("France", "Japan", "Canada", "Peru")
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(table)))

        optimizer = optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=2, max_labeled_demos=0)
        optimizer.compile(cot, trainset=trainset)

        demos = [demo.toDict() for demo in cot.demos]
        assert len(demos) == 2
        assert all(demo["augmented"] is True for demo in demos)
        # Only the metric-passing traces (France, Canada) became demos —
        # complete with the reasoning field the trace really produced.
        assert [demo["answer"] for demo in demos] == ["Paris", "Ottawa"]
        assert [demo["reasoning"] for demo in demos] == ["Recall France.", "Recall Canada."]

    def test_bootstrap_is_deterministic_under_seed(self):
        trainset = trainset_of("France", "Japan", "Canada", "Peru")

        def run(seed):
            cot = dspy.ChainOfThought("question -> answer")
            dspy.configure(lm=dspy.DummyLM(oracle(CAPITALS)))
            optim.BootstrapFewShot(
                metric=exact_answer, max_bootstrapped_demos=1, max_labeled_demos=3, seed=seed
            ).compile(cot, trainset=trainset)
            return [demo.toDict() for demo in cot.demos]

        assert run(0) == run(0)
        assert run(0) != run(3)  # the labeled fill samples differently

    def test_bootstrap_respects_both_demo_caps(self):
        trainset = trainset_of("France", "Japan", "Canada", "Peru")
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(CAPITALS)))

        optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=1, max_labeled_demos=3).compile(
            cot, trainset=trainset
        )

        augmented = [demo for demo in cot.demos if demo.get("augmented")]
        labeled = [demo for demo in cot.demos if not demo.get("augmented")]
        assert len(cot.demos) == 3
        assert len(augmented) == 1
        assert len(labeled) == 2
        # Labeled fill comes from examples the bootstrap did not consume.
        assert all("reasoning" not in demo._store for demo in labeled)

    def test_bootstrap_teacher_separation(self):
        trainset = trainset_of("France", "Canada")
        student = dspy.ChainOfThought("question -> answer")
        teacher_lm = dspy.DummyLM(oracle(CAPITALS))
        teacher = dspy.ChainOfThought("question -> answer", lm=teacher_lm)
        student_lm = dspy.DummyLM([])  # would refuse loudly if ever called
        dspy.configure(lm=student_lm)

        optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=2, max_labeled_demos=0).compile(
            student, trainset=trainset, teacher=teacher
        )

        assert len(student.demos) == 2  # gathered by the teacher's runs
        assert student_lm.calls == []  # the student's LM never fired
        assert teacher.demos == []  # the teacher's own state was restored

    def test_bootstrap_structure_mismatch_refuses(self):
        student = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        with pytest.raises(ValueError, match="disagree on the signature"):
            optim.BootstrapFewShot(metric=exact_answer).compile(
                student, trainset=trainset_of("France"), teacher=dspy.Predict("question -> answer")
            )
        with pytest.raises(ValueError, match="share the predictor instance"):
            optim.BootstrapFewShot(metric=exact_answer).compile(
                student, trainset=trainset_of("France"), teacher=student
            )


# ---------------------------------------------------------------------------
# BootstrapFewShot on ReAct: composite programs, tool leaves and all
# ---------------------------------------------------------------------------


def lookup(country: str) -> str:
    """Return the capital of a country from a fixed table."""
    capitals = {"france": "Paris", "canada": "Ottawa"}
    return capitals.get(country.lower(), "unknown")


def react_pilot(messages):
    """One scripted agent: look the country up, finish, extract from the observation."""
    system = str(messages[0]["content"])
    current = str(messages[-1]["content"])
    country = next((c for c in CAPITALS if c.lower() in current.lower()), None)
    if "next_tool_name" in system:
        if "observation_0" not in current:
            name = country.lower() if country else "atlantis"
            return chat_completion(
                next_thought="Look it up.",
                next_tool_name="lookup",
                next_tool_args=json.dumps({"country": name}),
            )
        return chat_completion(next_thought="Enough gathered.", next_tool_name="finish", next_tool_args="{}")
    answer = next((capital for capital in CAPITALS.values() if capital in current), "unknown")
    return chat_completion(reasoning="From the observation.", answer=answer)


class TestBootstrapReAct:
    def bootstrap(self):
        # Atlantis fails the metric (the tool knows no such country), so
        # only the France and Canada runs may produce demos.
        trainset = [make_example("France"), make_example("Atlantis", answer="Mythopolis"), make_example("Canada")]
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=3)
        dspy.configure(lm=dspy.DummyLM(react_pilot))
        optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=2, max_labeled_demos=0).compile(
            react, trainset=trainset
        )
        return react

    def test_react_bootstrap_collects_per_predictor_demos(self):
        react = self.bootstrap()

        assert len(react.react.demos) == 2
        assert len(react.extract.demos) == 2
        # Only metric-passing runs contributed; the last react call per
        # run is the finish step, the extract demo carries the answer.
        assert [demo.next_tool_name for demo in react.react.demos] == ["finish", "finish"]
        assert [demo.answer for demo in react.extract.demos] == ["Paris", "Ottawa"]
        assert all(demo.augmented for demo in react.react.demos + react.extract.demos)

        records = react.to_manifest()["components"]["3b_demos"]
        assert len(records["react"]) == 2
        assert len(records["extract"]) == 2

    def test_optimized_react_compiles_clean_and_equivalence_holds(self):
        react = self.bootstrap()

        manifest = react.to_manifest()  # compile-clean, demos and all
        assert manifest["components"]["5_forward"]["self"]["body"][1]["range"] == 3

        engine_lm = dspy.DummyLM(react_pilot)
        dspy.configure(lm=engine_lm)
        engine_prediction = react(question="What is the capital of France?")

        native_lm = dspy.DummyLM(react_pilot)
        dspy.configure(lm=native_lm)
        native_prediction = react.forward_native(question="What is the capital of France?")

        assert len(engine_lm.calls) == len(native_lm.calls)
        for engine_call, native_call in zip(engine_lm.calls, native_lm.calls, strict=True):
            assert engine_call["messages"] == native_call["messages"]
            assert engine_call["kwargs"] == native_call["kwargs"]
        assert engine_prediction.toDict() == native_prediction.toDict()
        assert engine_prediction.answer == "Paris"


# ---------------------------------------------------------------------------
# Random search: N candidates, argmax kept
# ---------------------------------------------------------------------------


class TestRandomSearch:
    def search(self, **kwargs):
        # copycat only answers when a demo carries the capital: zero-shot
        # scores 0.0, every demo-carrying candidate scores 1.0.
        trainset = trainset_of("France", "Canada")
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(copycat))
        optimizer = optim.BootstrapFewShotWithRandomSearch(
            metric=exact_answer, max_bootstrapped_demos=2, num_candidate_programs=1, **kwargs
        )
        best = optimizer.compile(cot, trainset=trainset, valset=trainset)
        return optimizer, best

    def test_random_search_picks_the_argmax_candidate(self):
        optimizer, best = self.search()

        scores = [entry["score"] for entry in optimizer.trajectory]
        assert scores == [0.0, 1.0, 1.0, 1.0]
        assert [entry["accepted"] for entry in optimizer.trajectory] == [True, True, False, False]
        assert [entry["label"] for entry in optimizer.trajectory] == [
            "zero-shot",
            "labeled-only",
            "bootstrap-unshuffled",
            "bootstrap-shuffle-0",
        ]

        # The returned program re-evaluates to the trajectory's max...
        result = optim.evaluate(best, trainset_of("France", "Canada"), exact_answer)
        assert result.score == max(scores)
        # ...and carries the FIRST best candidate's state (labeled-only:
        # no augmented demos), because acceptance is strict improvement.
        assert len(best.demos) == 2
        assert all(demo.get("augmented") is None for demo in best.demos)

    def test_random_search_stops_at_score(self):
        optimizer, _ = self.search(stop_at_score=1.0)
        assert [entry["score"] for entry in optimizer.trajectory] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# checkpoint == save: accepted candidates are loadable artifacts
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_random_search_checkpoints_are_loadable_artifacts(self, tmp_path):
        trainset = trainset_of("France", "Canada")
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(copycat))
        optimizer = optim.BootstrapFewShotWithRandomSearch(
            metric=exact_answer, max_bootstrapped_demos=2, num_candidate_programs=1
        )
        optimizer.compile(cot, trainset=trainset, valset=trainset, checkpoint_dir=tmp_path / "run")

        # Two accepted candidates (zero-shot, then labeled-only) landed
        # as artifact directories, with the trajectory in scores.json.
        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert [record["candidate"] for record in scores] == ["candidate-000", "candidate-001"]
        assert [record["score"] for record in scores] == [0.0, 1.0]
        assert [record["label"] for record in scores] == ["zero-shot", "labeled-only"]
        assert (tmp_path / "run" / "candidate-000" / "manifest.json").is_file()

        loaded = dspy.load(tmp_path / "run" / "candidate-001", bindings={"lm": {"dummy": dspy.DummyLM(copycat)}})
        replayed = loaded(question="What is the capital of France?")
        assert replayed.answer == "Paris"  # the checkpointed demos ride along

    def test_labeled_checkpoint_is_a_loadable_artifact(self, tmp_path):
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(CAPITALS)))
        optim.LabeledFewShot(k=2).compile(
            cot, trainset=trainset_of("France", "Canada"), checkpoint_dir=tmp_path / "run"
        )

        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert scores == [{"candidate": "candidate-000", "score": None, "label": "labeled-fewshot"}]

        loaded = dspy.load(
            tmp_path / "run" / "candidate-000", bindings={"lm": {"dummy": dspy.DummyLM(oracle(CAPITALS))}}
        )
        assert loaded(question="What is the capital of Canada?").answer == "Ottawa"


# ---------------------------------------------------------------------------
# The charter's round trip: augmented CoT demos survive save/load exactly
# ---------------------------------------------------------------------------


class TestDemoRoundTrip:
    def test_cot_augmented_demos_round_trip_save_load(self, tmp_path):
        trainset = trainset_of("France", "Canada")
        cot = dspy.ChainOfThought("question -> answer")
        dspy.configure(lm=dspy.DummyLM(oracle(CAPITALS)))
        optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=2, max_labeled_demos=0).compile(
            cot, trainset=trainset
        )
        assert all(demo.augmented and demo.reasoning for demo in cot.demos)

        native_lm = dspy.DummyLM(oracle(CAPITALS))
        dspy.configure(lm=native_lm)
        direct = cot(question="What is the capital of France?")

        target = tmp_path / "artifact"
        cot.save(target)
        records = json.loads((target / "manifest.json").read_text())["components"]["3b_demos"]["self"]
        assert [record["augmented"] for record in records] == [True, True]

        loaded_lm = dspy.DummyLM(oracle(CAPITALS))
        loaded = dspy.load(target, bindings={"lm": {"dummy": loaded_lm}})
        replayed = loaded(question="What is the capital of France?")

        assert replayed.toDict() == direct.toDict()
        # The rebuilt predictor renders the demos byte-for-byte.
        assert loaded_lm.calls[-1]["messages"] == native_lm.calls[-1]["messages"]
        rendered = "\n".join(str(message["content"]) for message in loaded_lm.calls[-1]["messages"])
        assert "Recall Canada." in rendered  # the augmented demo's reasoning rode along
