"""Model backends. Each takes a list of RGB frames and returns a parsed prediction.

    backend = make_backend("openai:gpt-4.1")
    pred = backend.predict(frames, dt=0.15)

Backends:
    openai:<model>     OpenAI Chat Completions with a strict JSON schema
    anthropic:<model>  Anthropic Messages API with structured output
    swap_rb:<spec>     wraps another backend and swaps red/blue before sending,
                       reproducing the colour bug in the original scripts
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
from PIL import Image

from .parse import parse_prediction
from .prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, user_text


def encode_png(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class Backend:
    name = "base"

    def _call(self, frames, dt) -> str:
        raise NotImplementedError

    def predict(self, frames, dt: float = 0.15) -> dict:
        t = time.time()
        text = self._call(frames, dt)
        pred = parse_prediction(text)
        pred["latency_s"] = round(time.time() - t, 3)
        pred["raw"] = text
        return pred


class OpenAIBackend(Backend):
    def __init__(self, model: str = "gpt-4.1"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def _call(self, frames, dt):
        content = [{"type": "text", "text": user_text(len(frames), dt)}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_png(f)}"}}
                    for f in frames]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "hip_prediction", "strict": True,
                                             "schema": OUTPUT_SCHEMA}},
            max_tokens=1000,
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend(Backend):
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    def _call(self, frames, dt):
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": encode_png(f)}}
                   for f in frames]
        content.append({"type": "text", "text": user_text(len(frames), dt)})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        if resp.stop_reason == "refusal":
            return ""
        return next((b.text for b in resp.content if b.type == "text"), "")


class SwapRBBackend(Backend):
    """Sends BGR frames as if they were RGB, exactly what the old scripts did."""

    def __init__(self, inner: Backend):
        self.inner = inner
        self.name = f"swap_rb:{inner.name}"

    def predict(self, frames, dt: float = 0.15) -> dict:
        return self.inner.predict([f[:, :, ::-1].copy() for f in frames], dt)


def make_backend(spec: str) -> Backend:
    kind, _, rest = spec.partition(":")
    if kind == "openai":
        return OpenAIBackend(rest or "gpt-4.1")
    if kind == "anthropic":
        return AnthropicBackend(rest or "claude-opus-5-5")
    if kind == "swap_rb":
        return SwapRBBackend(make_backend(rest))
    raise ValueError(f"unknown backend {spec!r}")
