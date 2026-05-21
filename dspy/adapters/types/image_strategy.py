"""Rendering strategies for `dspy.Image` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dspy.adapters.types.image import Image
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMImagePart, LMOutput, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeImage(TypeStrategy[Image]):
    """Render image fields as normalized image parts and parse image outputs."""

    marker_type: type[Image] = Image
    detail: Literal["low", "high", "auto"] | None = None

    def render_input(self, *, field_name: str, field: Any, value: Image, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_input_fields=(field_name,),
            user_parts=[
                LMTextPart(text=f"\n\n{field_name}:\n"),
                _image_value_to_lm_part(value, detail=self.detail),
            ],
        )

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_output_fields=(field_name,),
            system_parts=[
                LMTextPart(
                    text=(
                        f"When producing `{field_name}`, return it as a native image output part "
                        "if the backend supports image generation."
                    )
                )
            ],
        )

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> Image | None:
        if isinstance(output, LMOutput):
            if not output.images:
                return None
            return _lm_part_to_image(output.images[0])
        if isinstance(output, dict):
            value = output.get(field_name)
            if isinstance(value, Image):
                return value
            if isinstance(value, str):
                return Image(value)
        if isinstance(output, str):
            return Image(output.strip())
        return None


def _image_value_to_lm_part(image: Image, *, detail: Literal["low", "high", "auto"] | None = None) -> LMImagePart:
    source = image.url
    if source.startswith("data:"):
        header, data = source.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return LMImagePart(data=data, media_type=media_type, detail=detail)
    return LMImagePart(url=source, detail=detail)


def _lm_part_to_image(image: LMImagePart) -> Image:
    if image.url is not None:
        return Image(image.url)
    if image.data is not None:
        return Image(f"data:{image.media_type};base64,{image.data}")
    if image.file_id is not None:
        return Image({"url": image.file_id})
    if image.path is not None:
        return Image(image.path)
    raise ValueError("LMImagePart has no image source.")
