# dspy.Refine

`dspy.Refine` extends the best-of-N strategy with a *feedback loop*. After each below-threshold attempt, it analyses the execution trace, assigns blame to sub-modules, and injects concrete advice as a hint into the next retry — making each attempt more informed than the last.

<!-- START_API_REF -->
::: dspy.Refine
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

### Basic refinement

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

qa = dspy.ChainOfThought("question -> answer")

def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=qa, N=3, reward_fn=one_word_answer, threshold=1.0,
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)
```

### How the feedback loop works

1. **Attempt** — Run the module with a unique rollout ID at `temperature=1.0`
2. **Score** — Evaluate the prediction with `reward_fn`
3. **If below threshold** — Analyse the full execution trace and generate per-module advice via the internal `OfferFeedback` signature
4. **Retry** — Re-run the module with the generated advice injected as a `hint_` input field
5. **Repeat** until threshold is met or `N` attempts are exhausted
6. **Return** the best prediction across all attempts
