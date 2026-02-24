# dspy.BestOfN

`dspy.BestOfN` is an inference-time scaling strategy that runs a module up to *N* times (each with a distinct rollout at `temperature=1.0`) and returns the prediction with the highest reward, or the first one that meets the threshold.

<!-- START_API_REF -->
::: dspy.BestOfN
    handler: python
    options:
        members:
            - __call__
            - acall
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - inspect_history
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## Usage Examples

### Basic best-of-N selection

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

qa = dspy.ChainOfThought("question -> answer")

def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=qa, N=3, reward_fn=one_word_answer, threshold=1.0,
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)
```

## BestOfN vs Refine

| | `dspy.BestOfN` | `dspy.Refine` |
|---|---|---|
| Attempts are independent | ✅ | ❌ (feedback between attempts) |
| Cost per attempt | Lower | Higher (feedback generation) |
| Convergence speed | Depends on luck | Often faster (guided retries) |
| Use when | Diversity matters | Targeted improvement matters |
