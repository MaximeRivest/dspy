# Output Refinement: BestOfN and Refine

Both `BestOfN` and `Refine` are DSPy modules designed to improve the reliability and quality of predictions by making multiple `LM` calls with different rollout IDs to bypass caching. Both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.

## BestOfN

`BestOfN` is a module that runs the provided module multiple times (up to `N`) with different rollout IDs. It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meets the threshold.

### Basic Usage

Lets say we wanted to have the best chance of getting a one word answer from the model. We could use `BestOfN` to try multiple rollout IDs and return the best result.

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Error Handling

By default, if the module encounters an error during an attempt, it will continue trying until it reaches `N` attempts. You can adjust this behavior with the `fail_count` parameter:

```python
best_of_3 = dspy.BestOfN(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="What is the capital of Belgium?")
# raises an error after the first failure
```

## Refine

`Refine` extends the functionality of `BestOfN` by adding a feedback loop. After each unsuccessful attempt (except the final one), feedback about the module's performance becomes a hint for subsequent runs. The feedback can come from your own reward function, or — as a fallback — from an automatic LM critique of the failed attempt.

### Basic Usage

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Feedback-Bearing Rewards

A reward function often already knows *why* it scored an attempt low. Instead of a bare float, it can return `dspy.Prediction(score=..., feedback="...")` — the same score-plus-feedback shape GEPA metrics use. When feedback is provided, `Refine` uses it directly as the hint for the next attempt and skips the LM critic entirely:

```python
def one_word_answer(args, pred: dspy.Prediction):
    if len(pred.answer.split()) == 1:
        return dspy.Prediction(score=1.0, feedback="")
    return dspy.Prediction(score=0.0, feedback="The answer must be exactly one word.")

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)
```

### Feedback Policy

Whether the automatic LM critique runs is an explicit policy, not an inevitability. The `feedback_policy` parameter takes one of:

- `"auto"` (default): use the reward function's feedback when it provides some; otherwise fall back to an LM critic that inspects the failed attempt. This matches the historical behavior for float-returning rewards.
- `"eval"`: only use feedback the reward function itself provides; never invoke the LM critic.
- `"none"`: never hint; `Refine` degrades to resampling and picking the best attempt.

Hints are non-authoritative: they are delivered to the retry as an explicit `hint_` input field whose description names their source and tells the model it may ignore them. The winning attempt's trace is merged back without the hint, and the full per-attempt record (scores, feedback, and hints with provenance) is attached to the returned prediction as `prediction._refinement`.

### Error Handling

Like `BestOfN`, `Refine` will try up to `N` times by default, even if errors occur. You can control this with the `fail_count` parameter:

```python
# Stop after the first error
refine = dspy.Refine(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)
```

If every attempt fails, `Refine` raises the last error instead of returning `None`.

## Comparison: BestOfN vs. Refine

Both modules serve similar purposes but differ in their approach:

- `BestOfN` simply tries different rollout IDs and selects the best resulting prediction as defined by the `reward_fn`.
- `Refine` adds a feedback loop. When the `reward_fn` returns feedback alongside its score, that feedback becomes the hint for subsequent runs; when it returns only a score, `Refine` can fall back to using the LM to generate detailed feedback about the module's performance from the previous prediction and the code of the `reward_fn` (see `feedback_policy` above).

Both accept reward functions that return either a plain float or a score-bearing `dspy.Prediction`.

## Practical Examples

### Ensuring Factual Correctness

```python
import dspy

class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)
```

### Summarization - Controlling Response Length

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Reward the summary for being close to 75 words with a tapering off for longer summaries.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[Long text to summarize...]"
)
print(result.summary)
```

## Migration from `dspy.Suggest` and `dspy.Assert`

`BestOfN` and `Refine` are the replacements for `dspy.Suggest` and `dspy.Assert` as of DSPy 2.6.
