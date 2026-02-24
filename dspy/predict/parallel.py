import threading
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


class Parallel:
    """Execute multiple DSPy module calls in parallel using threads.

    ``Parallel`` takes a list of ``(module, inputs)`` pairs and runs them concurrently,
    returning a list of results in the same order. It is the building block behind
    ``dspy.Module.batch()`` and is useful whenever you need to fan out heterogeneous
    calls — different modules, different inputs, or both — across threads.

    Example:

        Basic usage with ``dspy.Predict``:

        ```python
        import dspy

        dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

        predict = dspy.Predict("question -> answer")
        parallel = dspy.Parallel(num_threads=4)

        examples = [
            dspy.Example(question="What is 1+1?").with_inputs("question"),
            dspy.Example(question="What is 2+2?").with_inputs("question"),
        ]

        results = parallel([(predict, ex) for ex in examples])
        ```

        Mixing different modules:

        ```python
        cot = dspy.ChainOfThought("question -> answer")
        predict = dspy.Predict("question -> answer")

        results = parallel([
            (cot,     dspy.Example(question="Why is the sky blue?").with_inputs("question")),
            (predict, dspy.Example(question="What is 2+2?").with_inputs("question")),
        ])
        ```

        Collecting failed examples:

        ```python
        parallel = dspy.Parallel(num_threads=2, return_failed_examples=True, provide_traceback=True)
        results, failed_examples, exceptions = parallel([(predict, ex) for ex in examples])
        ```

        Nested parallel execution:

        ```python
        inner_parallel = dspy.Parallel(num_threads=2)
        outer_parallel = dspy.Parallel(num_threads=2)

        results = outer_parallel([
            (predict, example1),
            (inner_parallel, [           # nested: a Parallel as the "module"
                (predict, example2),
                (predict, example3),
            ]),
        ])
        ```

    Args:
        num_threads: Maximum number of worker threads. Defaults to ``dspy.settings.num_threads``.
        max_errors: Maximum number of errors to tolerate before aborting execution.
            If ``None``, inherits from ``dspy.settings.max_errors``.
        access_examples: When ``True`` (default), ``dspy.Example`` inputs are unpacked
            via ``example.inputs()`` before being passed to the module. Set to ``False``
            to pass the ``Example`` object directly.
        return_failed_examples: When ``True``, ``forward`` returns a 3-tuple
            ``(results, failed_examples, exceptions)`` instead of just the results list.
        provide_traceback: Whether to include full tracebacks in error logs.
            If ``None``, inherits from ``dspy.settings.provide_traceback``.
        disable_progress_bar: Disable the ``tqdm`` progress bar.
        timeout: Seconds to wait before resubmitting a straggler task. Set to ``0``
            to disable straggler resubmission.
        straggler_limit: Only check for stragglers when this many or fewer tasks
            remain unfinished.

    Note:
        **Input types:** Each pair in ``exec_pairs`` is ``(module, inputs)`` where
        ``inputs`` can be a ``dspy.Example``, a ``dict`` (keyword arguments), a
        ``tuple`` (positional arguments), or a ``list`` when the module is itself a
        ``Parallel`` instance (nested execution).

        **Relationship to Module.batch:** ``dspy.Module.batch()`` is a convenience
        wrapper that creates a ``Parallel`` instance internally. Use ``Parallel``
        directly when you need to mix different modules in a single fan-out or want
        to nest parallel calls.

        **Thread safety:** Each worker thread receives an isolated copy of
        ``dspy.settings`` overrides, so context managers like ``dspy.context(lm=...)``
        in the calling thread are propagated correctly.

        **Error handling:** Failed tasks produce ``None`` in the results list. Use
        ``return_failed_examples=True`` to collect the failing inputs and their
        exceptions for inspection or retry.
    """

    def __init__(
        self,
        num_threads: int | None = None,
        max_errors: int | None = None,
        access_examples: bool = True,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
        straggler_limit: int = 3,
    ):
        super().__init__()
        self.num_threads = num_threads or settings.num_threads
        self.max_errors = settings.max_errors if max_errors is None else max_errors
        self.access_examples = access_examples
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar
        self.timeout = timeout
        self.straggler_limit = straggler_limit

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.failed_examples = []
        self.exceptions = []

    def forward(self, exec_pairs: list[tuple[Any, Example]], num_threads: int | None = None) -> list[Any]:
        """Execute a list of ``(module, inputs)`` pairs in parallel.

        Args:
            exec_pairs: A list of ``(module, inputs)`` tuples. Each *module* is a
                callable (typically a ``dspy.Module``) and each *inputs* is one of:

                - ``dspy.Example`` — unpacked via ``example.inputs()`` when
                  ``access_examples=True``, or passed directly otherwise.
                - ``dict`` — passed as keyword arguments.
                - ``tuple`` — passed as positional arguments.
                - ``list`` — used for nested ``Parallel`` execution.
            num_threads: Override the instance-level ``num_threads`` for this call.

        Returns:
            A list of results in the same order as ``exec_pairs``. Failed tasks
            appear as ``None``. If ``return_failed_examples=True``, returns a 3-tuple
            ``(results, failed_examples, exceptions)`` instead.
        """
        num_threads = num_threads if num_threads is not None else self.num_threads

        executor = ParallelExecutor(
            num_threads=num_threads,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            disable_progress_bar=self.disable_progress_bar,
            timeout=self.timeout,
            straggler_limit=self.straggler_limit,
        )

        def process_pair(pair):
            result = None
            module, example = pair

            if isinstance(example, Example):
                if self.access_examples:
                    result = module(**example.inputs())
                else:
                    result = module(example)
            elif isinstance(example, dict):
                result = module(**example)
            elif isinstance(example, list) and module.__class__.__name__ == "Parallel":
                result = module(example)
            elif isinstance(example, tuple):
                result = module(*example)
            else:
                raise ValueError(
                    f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
                )
            return result

        # Execute the processing function over the execution pairs
        results = executor.execute(process_pair, exec_pairs)

        # Populate failed examples and exceptions from the executor
        if self.return_failed_examples:
            for failed_idx in executor.failed_indices:
                if failed_idx < len(exec_pairs):
                    _, original_example = exec_pairs[failed_idx]
                    self.failed_examples.append(original_example)
                    if exception := executor.exceptions_map.get(failed_idx):
                        self.exceptions.append(exception)

            return results, self.failed_examples, self.exceptions
        else:
            return results

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
