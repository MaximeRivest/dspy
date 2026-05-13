"""Feature reporting for normalized DSPy language models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

FeatureState = Literal["unknown", "observed", "inferred", "unsupported"]
ReportFormat = Literal["text", "json"]
SupportState = Literal["supported", "unsupported", "unknown"]

REQUEST_FEATURES = (
    "text",
    "input_image",
    "input_audio",
    "input_file",
    "tools",
    "tool_choice",
    "assistant_tool_calls",
    "tool_results",
    "response_schema",
    "reasoning_config",
    "prompt_cache",
    "logprobs",
    "multiple_outputs",
    "provider_extensions",
)

RESPONSE_FEATURES = (
    "text",
    "reasoning",
    "tool_calls",
    "citations",
    "output_image",
    "output_audio",
    "output_file",
    "refusal",
    "usage",
    "cost",
)


class FeatureReport(str):
    """A text feature report that displays cleanly in notebooks."""

    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))


@dataclass(frozen=True)
class FeatureStatus:
    """Status and evidence for one DSPy LM integration feature.

    A status is truthy when DSPy has observed or inferred support. Use the
    `status` and `evidence` fields when you need to explain why a feature is
    available, unavailable, or still unknown.

    Args:
        name: Feature name, such as `usage` or `request.input_image`.
        status: Current support state for the feature.
        evidence: Short explanation for the current state.
    """

    name: str
    status: FeatureState
    evidence: str | None = None

    def __bool__(self) -> bool:
        return self.status in {"observed", "inferred"}


@dataclass(frozen=True)
class LMSupportIssue:
    """One unsupported or unknown feature required by an LM request."""

    feature: str
    message: str
    status: FeatureState


@dataclass(frozen=True)
class LMRequestSupport:
    """Support report for one normalized LM request."""

    status: SupportState
    issues: list[LMSupportIssue]

    def __bool__(self) -> bool:
        return self.status == "supported"


class LMFeatureGroup:
    """Expose request or response feature statuses as attributes."""

    def __init__(self, reporter: LMFeatureReporter, namespace: Literal["request", "response"]):
        self._reporter = reporter
        self._namespace = namespace

    @property
    def names(self) -> tuple[str, ...]:
        return REQUEST_FEATURES if self._namespace == "request" else RESPONSE_FEATURES

    def __getattr__(self, name: str) -> FeatureStatus:
        if name in self.names:
            return self.status(name)
        raise AttributeError(f"{type(self).__name__!r} has no feature {name!r}.")

    def status(self, name: str) -> FeatureStatus:
        return self._reporter.status(f"{self._namespace}.{name}")

    def supports(self, name: str) -> bool:
        return bool(self.status(name))

    def to_dict(self) -> dict[str, dict[str, str | None]]:
        return {
            name: {"status": (feature := self.status(name)).status, "evidence": feature.evidence} for name in self.names
        }


class LMFeatureReporter:
    """Report implementation support and observed LM integration features.

    The reporter combines method introspection, request/response shape support,
    and observations from completed `LMResponse` objects. Request support is
    about the `LanguageModel` implementation: whether it can faithfully map a
    normalized DSPy request shape. It is intentionally separate from
    `lm.capabilities`, which describes model- or deployment-specific native
    abilities such as whether one concrete model can use tools, images, or
    reasoning.

    Args:
        lm: Language model instance whose features are being reported.

    Examples:
        ```python
        if lm.features.request.input_image:
            response = lm("Describe this image", image)
        else:
            print(lm.features.request.input_image.evidence)
        ```
    """

    _KNOWN_FEATURES = (
        "text_generation",
        "usage",
        "cost",
        "streaming",
        "async_streaming",
        "native_async",
        "caching",
        "history",
        "context_window_errors",
    )

    def __init__(self, lm: Any):
        self._lm = lm
        self._observed: dict[str, FeatureStatus] = {}
        self.request = LMFeatureGroup(self, "request")
        self.response = LMFeatureGroup(self, "response")

    def __getattr__(self, name: str) -> FeatureStatus:
        if name in self._KNOWN_FEATURES:
            return self.status(name)
        raise AttributeError(f"{type(self).__name__!r} has no feature {name!r}.")

    @property
    def names(self) -> tuple[str, ...]:
        """Known DSPy LM integration feature names."""
        return self._KNOWN_FEATURES

    def observe(self, name: str, evidence: str | None = None) -> None:
        """Record that a feature worked at runtime."""
        self._observed[name] = FeatureStatus(name=name, status="observed", evidence=evidence)

    def observe_error(self, error: Exception) -> None:
        """Record features shown by a normalized LM error."""
        from dspy.utils.exceptions import ContextWindowExceededError

        if isinstance(error, ContextWindowExceededError):
            self.observe(
                "context_window_errors",
                "This LM normalizes context-window failures to ContextWindowExceededError.",
            )

    def supports(self, name: str) -> bool:
        """Return whether a feature is observed or inferred as supported."""
        return bool(self.status(name))

    def status(self, name: str) -> FeatureStatus:
        """Return the current status for one feature.

        Names may be flat integration features like `usage`, or namespaced
        request/response features like `request.input_image`.
        """
        if name in self._observed:
            return self._observed[name]

        if name.startswith("request."):
            return self._request_status(name.removeprefix("request."))
        if name.startswith("response."):
            return self._response_status(name.removeprefix("response."))

        if name == "text_generation":
            return self._method_status(name, "forward")
        if name == "native_async":
            return self._method_status(name, "aforward")
        if name == "streaming":
            support = getattr(self._lm, "support", None) if hasattr(self._lm, "_support") else None
            if support is not None and getattr(support, "streaming", False):
                return FeatureStatus(name=name, status="inferred", evidence="Declared in lm.support.")
            return self._method_status(name, "forward_stream")
        if name == "async_streaming":
            support = getattr(self._lm, "support", None) if hasattr(self._lm, "_support") else None
            if support is not None and getattr(support, "async_streaming", False):
                return FeatureStatus(name=name, status="inferred", evidence="Declared in lm.support.")
            return self._method_status(name, "aforward_stream")
        if name == "caching":
            return self._caching_status()
        if name == "history":
            return FeatureStatus(
                name=name,
                status="inferred",
                evidence="LanguageModel records LMRequest and LMResponse unless history is disabled.",
            )
        if name == "usage":
            return self._response_status("usage", flat_name="usage")
        if name == "cost":
            return self._response_status("cost", flat_name="cost")
        if name == "context_window_errors":
            return FeatureStatus(
                name=name,
                status="unknown",
                evidence="No normalized context-window error has been observed yet.",
            )

        return FeatureStatus(name=name, status="unknown", evidence="Unknown DSPy LM feature.")

    def explain(self, name: str) -> str | None:
        """Return the evidence string for one feature."""
        return self.status(name).evidence

    def validate_request(self, request: Any) -> LMRequestSupport:
        """Return whether this implementation supports every feature used by `request`."""
        issues = []
        for feature_name in _required_request_features(request):
            feature = self.request.status(feature_name)
            if not feature:
                issues.append(
                    LMSupportIssue(
                        feature=f"request.{feature_name}",
                        status=feature.status,
                        message=feature.evidence or f"request.{feature_name} is not supported.",
                    )
                )

        issues.extend(_rich_support_issues(request, getattr(self._lm, "support", None) if hasattr(self._lm, "_support") else None))

        if any(issue.status == "unsupported" for issue in issues):
            status: SupportState = "unsupported"
        elif issues:
            status = "unknown"
        else:
            status = "supported"
        return LMRequestSupport(status=status, issues=issues)

    def report(self, *, format: ReportFormat = "text") -> str | dict[str, Any]:
        """Return a feature report for this LM."""
        data = self._report_data()
        if format == "json":
            return data
        if format != "text":
            raise ValueError("format must be 'text' or 'json'.")
        return self._report_text(data)

    def to_json(self, **kwargs: Any) -> str:
        """Return the feature report as a JSON string."""
        options = {"indent": 2, **kwargs}
        return json.dumps(self.report(format="json"), **options)

    def _request_status(self, name: str) -> FeatureStatus:
        declared = self._declared_request_status(name)
        if declared is not None:
            return declared
        support = getattr(self._lm, "get_request_feature_statuses", lambda: {})()
        feature = support.get(name)
        if feature is not None:
            return feature
        return FeatureStatus(
            name=f"request.{name}",
            status="unknown",
            evidence=f"No request support status is registered for {name}.",
        )

    def _declared_request_status(self, name: str) -> FeatureStatus | None:
        if not hasattr(self._lm, "_support"):
            return None
        support = getattr(self._lm, "support", None)
        if support is None:
            return None
        table = {
            "text": bool(getattr(support, "text", False)),
            "input_image": getattr(support, "images", None) is not None,
            "input_audio": bool(getattr(support, "audio", False)),
            "input_file": bool(getattr(support, "files", False)),
            "tools": getattr(support, "tools", None) is not None and bool(getattr(support.tools, "schemas", False)),
            "tool_choice": getattr(support, "tools", None) is not None and bool(getattr(support.tools, "schemas", False)),
            "assistant_tool_calls": getattr(support, "tools", None) is not None and bool(getattr(support.tools, "calls", False)),
            "tool_results": getattr(support, "tools", None) is not None and bool(getattr(support.tools, "results", False)),
            "response_schema": bool(getattr(support, "response_schema", False)),
            "reasoning_config": bool(getattr(support, "reasoning", False)),
            "prompt_cache": bool(getattr(support, "prompt_cache", False)),
            "logprobs": bool(getattr(support, "logprobs", False)),
            "multiple_outputs": bool(getattr(support, "multiple_outputs", False)),
            "provider_extensions": bool(getattr(support, "provider_extensions", False)),
        }
        if name not in table:
            return None
        ok = table[name]
        if ok:
            return FeatureStatus(f"request.{name}", "inferred", f"Declared in {type(self._lm).__name__}.support.")
        return FeatureStatus(f"request.{name}", "unsupported", f"{type(self._lm).__name__}.support does not declare request.{name}.")

    def _response_status(self, name: str, *, flat_name: str | None = None) -> FeatureStatus:
        observed_name = flat_name or f"response.{name}"
        if observed_name in self._observed:
            return self._observed[observed_name]
        declared = None if name in {"usage", "cost"} else self._declared_response_status(name, flat_name=flat_name)
        if declared is not None:
            return declared
        support = getattr(self._lm, "get_response_feature_statuses", lambda: {})()
        feature = support.get(name)
        if feature is not None:
            if flat_name is not None:
                return FeatureStatus(name=flat_name, status=feature.status, evidence=feature.evidence)
            return feature
        return FeatureStatus(
            name=flat_name or f"response.{name}",
            status="unknown",
            evidence=f"No response support status is registered for {name}.",
        )

    def _declared_response_status(self, name: str, *, flat_name: str | None = None) -> FeatureStatus | None:
        if not hasattr(self._lm, "_support"):
            return None
        support = getattr(self._lm, "support", None)
        if support is None:
            return None
        table = {
            "text": bool(getattr(support, "text", False)),
            "reasoning": bool(getattr(support, "reasoning", False)),
            "tool_calls": getattr(support, "tools", None) is not None and bool(getattr(support.tools, "calls", False)),
            "citations": bool(getattr(support, "citations", False)),
            "output_image": bool(getattr(support, "output_images", False)),
            "output_audio": bool(getattr(support, "output_audio", False)),
            "output_file": bool(getattr(support, "output_files", False)),
            "refusal": bool(getattr(support, "refusal", False)),
            "usage": bool(getattr(support, "usage", False)),
        }
        if name not in table:
            return None
        feature_name = flat_name or f"response.{name}"
        ok = table[name]
        if ok:
            return FeatureStatus(feature_name, "inferred", f"Declared in {type(self._lm).__name__}.support.")
        return FeatureStatus(feature_name, "unsupported", f"{type(self._lm).__name__}.support does not declare response.{name}.")

    def _report_data(self) -> dict[str, Any]:
        return {
            "model": getattr(self._lm, "model", None),
            "features": {
                name: {
                    "status": (feature := self.status(name)).status,
                    "evidence": feature.evidence,
                }
                for name in self._KNOWN_FEATURES
            },
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
        }

    def _report_text(self, data: dict[str, Any]) -> str:
        lines = [f"DSPy LM feature report for {data['model']}"]
        lines.extend(self._feature_section("Integration features", data["features"]))
        lines.extend(self._feature_section("Request support", data["request"], include_status=True))
        lines.extend(self._feature_section("Response support", data["response"], include_status=True))
        return FeatureReport("\n".join(lines))

    def _feature_section(
        self,
        title: str,
        features: dict[str, dict[str, Any]],
        *,
        include_status: bool = False,
    ) -> list[str]:
        lines = ["", f"{title}:"]
        if include_status:
            for name, feature in features.items():
                lines.append(f"  {name:<24} {feature['status']:<11} {feature['evidence'] or ''}")
            return lines

        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "observed": [],
            "inferred": [],
            "unsupported": [],
            "unknown": [],
        }
        for name, feature in features.items():
            grouped[feature["status"]].append((name, feature))
        for group_title, key in (
            ("Observed", "observed"),
            ("Inferred", "inferred"),
            ("Unsupported", "unsupported"),
            ("Unknown", "unknown"),
        ):
            if not grouped[key]:
                continue
            lines.extend([f"  {group_title}:"])
            for name, feature in grouped[key]:
                lines.append(f"    {name:<22} {feature['evidence'] or ''}")
        return lines

    def _method_status(self, name: str, method_name: str) -> FeatureStatus:
        base_method = self._base_language_model_method(method_name)
        current_method = getattr(type(self._lm), method_name, None)
        if current_method is not None and base_method is not None and current_method is not base_method:
            return FeatureStatus(name=name, status="inferred", evidence=f"{method_name}() is overridden.")
        return FeatureStatus(name=name, status="unsupported", evidence=f"{method_name}() is not overridden.")

    def _base_language_model_method(self, method_name: str) -> Any | None:
        for cls in type(self._lm).__mro__[1:]:
            if cls.__name__ == "LanguageModel":
                return getattr(cls, method_name, None)
        return None

    def _caching_status(self) -> FeatureStatus:
        enabled = bool(getattr(self._lm, "cache", False))
        suffix = "enabled" if enabled else "disabled"
        return FeatureStatus(
            name="caching",
            status="inferred",
            evidence=f"LanguageModel applies DSPy request caching before forward(); this instance has cache {suffix} by default.",
        )


def feature_status(name: str, status: FeatureState, evidence: str) -> FeatureStatus:
    """Create a `FeatureStatus` with a concise call site."""
    return FeatureStatus(name=name, status=status, evidence=evidence)


def _rich_support_issues(request: Any, support: Any) -> list[LMSupportIssue]:
    if support is None or getattr(support, "images", None) is None:
        return []
    image_support = support.images
    image_locations = []
    messages = getattr(request, "messages", []) or []
    latest_user_index = None
    for idx, message in enumerate(messages):
        if getattr(message, "role", None) == "user":
            latest_user_index = idx
    for idx, message in enumerate(messages):
        for part in getattr(message, "parts", []) or []:
            if getattr(part, "type", None) == "image":
                image_locations.append((idx, part))
    issues: list[LMSupportIssue] = []
    if getattr(image_support, "placement", None) == "latest_user_message_only":
        for idx, _part in image_locations:
            if idx != latest_user_index:
                issues.append(
                    LMSupportIssue(
                        feature="request.input_image",
                        status="unsupported",
                        message="This request contains images outside the latest user message, but lm.support.images.placement is 'latest_user_message_only'.",
                    )
                )
                break
    for _idx, part in image_locations:
        if getattr(part, "url", None) is not None and not getattr(image_support, "urls", False):
            issues.append(LMSupportIssue("request.input_image", "Image URLs are not supported by lm.support.images.", "unsupported"))
        if getattr(part, "data", None) is not None and not getattr(image_support, "base64", False):
            issues.append(LMSupportIssue("request.input_image", "Base64 image data is not supported by lm.support.images.", "unsupported"))
        if getattr(part, "file_id", None) is not None and not getattr(image_support, "file_ids", False):
            issues.append(LMSupportIssue("request.input_image", "Image file IDs are not supported by lm.support.images.", "unsupported"))
        if getattr(part, "path", None) is not None and not getattr(image_support, "paths", False):
            issues.append(LMSupportIssue("request.input_image", "Image paths are not supported by lm.support.images.", "unsupported"))
        media_types = getattr(image_support, "media_types", None)
        if media_types is not None and getattr(part, "media_type", None) not in media_types:
            issues.append(LMSupportIssue("request.input_image", f"Image media type {getattr(part, 'media_type', None)!r} is not supported.", "unsupported"))
    return issues


def _required_request_features(request: Any) -> set[str]:
    required = {"text"}
    if getattr(request, "tools", None):
        required.add("tools")

    config = getattr(request, "config", None)
    if getattr(config, "response_format", None) is not None:
        required.add("response_schema")
    if getattr(config, "reasoning", None) is not None:
        required.add("reasoning_config")
    if getattr(config, "tool_choice", None) is not None:
        required.add("tool_choice")
    if getattr(config, "prompt_cache", None) is not None:
        required.add("prompt_cache")
    if getattr(config, "logprobs", None) is not None:
        required.add("logprobs")
    n = getattr(config, "n", None)
    if n is not None and n != 1:
        required.add("multiple_outputs")
    if getattr(config, "extensions", None):
        required.add("provider_extensions")

    for message in getattr(request, "messages", []) or []:
        for part in getattr(message, "parts", []) or []:
            part_type = getattr(part, "type", None)
            if part_type == "image":
                required.add("input_image")
            elif part_type == "audio":
                required.add("input_audio")
            elif part_type == "file":
                required.add("input_file")
            elif part_type == "tool_call":
                required.add("assistant_tool_calls")
            elif part_type == "tool_result":
                required.add("tool_results")
    return required
