"""LLM provider: Groq chat completions.

Groq serves open models (Llama 3.x, etc.) behind a fast OpenAI-style chat API.
We keep a tiny abstraction so the model/provider can be swapped without touching
the RAG layer. Supports both a blocking ``generate`` and a token ``generate_stream``,
plus ``describe_image`` for captioning images at ingest time.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Iterator

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_VISION_SYSTEM = (
    "You describe images for a search index. State what the image shows: its "
    "type (photo, chart, diagram, screenshot), the objects or people present, "
    "and any data, labels, or readings you can see. For charts, report the "
    "trend and the actual values. Be specific and factual; never guess at "
    "detail that is not visible."
)

_VISION_PROMPT = "Describe this image."


class GroqProvider:
    name = "groq"

    def __init__(self, settings: Settings | None = None) -> None:
        from groq import Groq

        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(get one at https://console.groq.com/keys)."
            )
        self.client = Groq(api_key=self.settings.groq_api_key)
        self.model = self.settings.groq_model

    def _messages(self, system: str, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
        )
        if self.settings.log_queries and getattr(resp, "usage", None):
            u = resp.usage
            logger.info(
                "groq.generate model=%s prompt_tokens=%s completion_tokens=%s",
                self.model, u.prompt_tokens, u.completion_tokens,
            )
        return resp.choices[0].message.content or ""

    def _encode_image(self, data: bytes) -> str:
        """Normalise arbitrary image bytes to a base64 JPEG the API accepts.

        Tesseract reads formats Groq won't (BMP, TIFF), and a phone photo is
        both larger than the request limit and slower than it needs to be, so
        everything is flattened to RGB, downscaled, and re-encoded.
        """
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, "white")
                flat.paste(rgba, mask=rgba.split()[-1])  # transparency -> white
                img = flat
            else:
                img = img.convert("RGB")
            limit = self.settings.vision_max_pixels
            if max(img.size) > limit:
                img.thumbnail((limit, limit))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def describe_image(self, data: bytes) -> str:
        """Caption an image so its visual content becomes searchable text."""
        b64 = self._encode_image(data)
        resp = self.client.chat.completions.create(
            model=self.settings.groq_vision_model,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=self.settings.vision_max_tokens,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield answer text chunks as they arrive from Groq."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, prompt),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            # Some SDK/model combinations attach usage to the final chunk.
            usage = getattr(chunk, "usage", None)
            if usage and self.settings.log_queries:
                logger.info(
                    "groq.stream model=%s prompt_tokens=%s completion_tokens=%s",
                    self.model, usage.prompt_tokens, usage.completion_tokens,
                )


def get_provider(settings: Settings | None = None) -> GroqProvider:
    return GroqProvider(settings)
