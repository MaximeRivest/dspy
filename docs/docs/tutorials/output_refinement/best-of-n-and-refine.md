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

`Refine` extends the functionality of `BestOfN` with an *explicit* feedback loop. A scalar reward
says that an attempt failed, not why, so `Refine` no longer manufactures an explanation
automatically. Instead, you declare both halves of the loop yourself:

1. **Where advice may go.** Give a predictor's Signature an input field named `hint` (configurable
   via `hint_field`). Only predictors that declare the field can receive advice; everything else is
   committed code that `Refine` resamples but never alters. If no predictor declares the field,
   feedback is never generated and `Refine` degrades to pure resampling.
2. **Where advice comes from.** Pass `feedback_fn(inputs, prediction, reward)`, returning `None`, a
   hint string, a `{predictor_name: advice}` dict, a `dspy.Hint`, or a list of these.

Hints are non-authoritative and scoped to the next attempt: they are delivered by filling the
declared hint field's default on a per-attempt copy of your module, so a value your own code passes
for that field always wins, and your configured adapter runs unmodified. Without `feedback_fn`,
`Refine` behaves exactly like `BestOfN`.

### Basic Usage

```python
import dspy

class HintedQA(dspy.Signature):
    question: str = dspy.InputField()
    hint: str = dspy.InputField(default="", desc="Advice from an earlier attempt; may be ignored.")
    answer: str = dspy.OutputField()

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

def one_word_feedback(args, pred: dspy.Prediction, reward: float):
    if reward < 1.0:
        return f"Your previous answer had {len(pred.answer.split())} words. Answer with exactly one word."

refine = dspy.Refine(
    module=dspy.ChainOfThought(HintedQA),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0,
    feedback_fn=one_word_feedback,
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Opting into an LM critic

If you want an LM to write the hints, pass `dspy.LMCritic`. The critic requires you to declare the
`objective` — what a good output looks like, in your own words — because it will not guess by
reading your reward function's source code. It sees only your declared objective, each module's
declared signature, and the evidence of the failed run (inputs, trajectory, outputs, reward):

```python
refine = dspy.Refine(
    module=dspy.ChainOfThought(HintedQA),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0,
    feedback_fn=dspy.LMCritic(objective="The answer must be a single word."),
)
```

### Inspecting what refinement did

The returned prediction carries a `RefinementReport` on its `_refinement` attribute: per-attempt
rollout IDs, rewards, applied and ignored hints (each with its provenance), and errors. The report
is metadata about the search, not program output, so it never appears in the prediction's fields,
in saved state, or in the trace.

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

## Comparison: BestOfN vs. Refine

Both modules serve similar purposes but differ in their approach:

- `BestOfN` simply tries different rollout IDs and selects the best resulting prediction as defined by the `reward_fn`.
- `Refine` adds an explicit feedback loop: your `feedback_fn` (or an opt-in `dspy.LMCritic`) turns a failed attempt into hints, which are delivered to the predictors that declare a `hint` input field on the next attempt. Without `feedback_fn`, `Refine` is `BestOfN`.

Earlier versions of `Refine` generated feedback automatically by sending the program's and the
reward function's source code to an internal LM critic, and injected the advice by appending a
hidden `hint_` field at the adapter boundary. Both behaviors are gone: feedback is now something
you declare, and hints only flow into fields your signatures declare.

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

class HintedQA(dspy.Signature):
    question: str = dspy.InputField()
    hint: str = dspy.InputField(default="", desc="Advice from an earlier attempt; may be ignored.")
    answer: str = dspy.OutputField()

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought(HintedQA),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0,
    feedback_fn=dspy.LMCritic(objective="The answer must be factually accurate."),
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
