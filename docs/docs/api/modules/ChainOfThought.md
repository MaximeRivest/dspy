# dspy.ChainOfThought

`dspy.ChainOfThought` wraps `dspy.Predict` by automatically prepending a `reasoning` output field. The language model is prompted to *"think step by step"* before producing the final answer, which often improves accuracy on tasks that benefit from intermediate reasoning.

<!-- START_API_REF -->
::: dspy.ChainOfThought
    handler: python
    options:
        members:
            - __call__
            - acall
            - aforward
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

### Basic chain-of-thought

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 15 * 7?")
print(result.reasoning)  # step-by-step reasoning
print(result.answer)     # final answer
```

### With a typed signature

```python
class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(QA)
result = cot(question="Why is the sky blue?")
```

### Predict vs ChainOfThought

| | `dspy.Predict` | `dspy.ChainOfThought` |
|---|---|---|
| Reasoning field | ❌ | ✅ (auto-prepended) |
| Token usage | Lower | Higher (reasoning tokens) |
| Accuracy on complex tasks | Baseline | Often improved |
| Drop-in replacement | — | ✅ Same call signature |
