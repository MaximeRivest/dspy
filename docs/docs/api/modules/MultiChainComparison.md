# dspy.MultiChainComparison

`dspy.MultiChainComparison` takes multiple chain-of-thought reasoning attempts and asks the language model to compare them holistically, producing a refined final answer. This is a simple self-consistency / refinement strategy that often improves accuracy over a single reasoning chain.

<!-- START_API_REF -->
::: dspy.MultiChainComparison
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

### Basic multi-chain comparison

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

# Generate 3 candidate reasoning chains
cot = dspy.ChainOfThought("question -> answer", n=3, temperature=0.7)
compare = dspy.MultiChainComparison("question -> answer", M=3)

# Get completions and compare
completions = cot(question="What is 18 * 7?").completions
result = compare(completions, question="What is 18 * 7?")
print(result.rationale)  # corrected reasoning
print(result.answer)     # refined answer
```

### How it works

1. Generate `M` diverse reasoning chains (e.g. using `ChainOfThought` with `n=M`)
2. Pass the `.completions` object to `MultiChainComparison`
3. The model sees all `M` attempts as numbered "Student Attempts" and produces a corrected rationale and final answer

!!! note
    The `M` parameter must match the number of completions passed to `forward`. Set `n=M` on your `ChainOfThought` or `Predict` call.
