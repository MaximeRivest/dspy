# Exceptions

DSPy exposes structured exception classes for DSPy and LM integrations. Provider-specific error mapping can vary by client; custom integrations can use these classes to expose consistent metadata.

<!-- START_API_REF -->
::: dspy.utils.exceptions
    handler: python
    options:
        members:
            - DSPyError
            - LMError
            - LMTransportError
            - LMConfigurationError
            - LMNotConfiguredError
            - LMUnsupportedFeatureError
            - LMProviderError
            - LMAuthError
            - LMBillingError
            - LMRateLimitError
            - LMInvalidRequestError
            - ContextWindowExceededError
            - LMUnsupportedModelError
            - LMTimeoutError
            - LMServerError
        show_source: true
        show_root_heading: false
        heading_level: 2
        docstring_style: google
        show_object_full_path: false
        separate_signature: false
<!-- END_API_REF -->
