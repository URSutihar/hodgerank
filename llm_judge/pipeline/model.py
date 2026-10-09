"""Ollama vision client for Qwen3.8-27B, plus a mock backend for testing.

Judgement is forced through a JSON schema so the model cannot answer with prose,
which is the usual source of unparseable VLM output.
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .config import Config

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are judging apparent age from photographs. "
    "Answer only with the requested JSON."
)

USER = (
    "Two photographs are shown: IMAGE 1 first, then IMAGE 2.\n\n"
    "Which person looks OLDER?\n\n"
    'Reply with JSON only: {"older": 1 or 2, "confidence": 0.0 to 1.0}\n'
    "  older      — 1 if the person in IMAGE 1 looks older, 2 if IMAGE 2 does\n"
    "  confidence — how sure you are\n"
    "You must pick one. Do not answer with a tie."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "older": {"type": "integer", "enum": [1, 2]},
        "confidence": {"type": "number"},
    },
    "required": ["older"],
}


def encode_image(path: Path, max_px: int = 512) -> str:
    """Base64 JPEG, downscaled so we do not waste tokens on upscaled faces."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_px:
            im.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaVisionJudge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.url = f"{cfg.host}/api/chat"
        self.session = requests.Session()

    # ── availability ────────────────────────────────────────────────────
    def health(self) -> tuple[bool, str]:
        try:
            r = self.session.get(f"{self.cfg.host}/api/tags", timeout=10)
            r.raise_for_status()
        except Exception as exc:                       # noqa: BLE001
            return False, (
                f"cannot reach Ollama at {self.cfg.host} ({exc}).\n"
                "  Start it with:  ollama serve")
        names = [m.get("name", "") for m in r.json().get("models", [])]
        want = self.cfg.model_name
        if want in names or any(n.split(":")[0] == want.split(":")[0] for n in names):
            return True, f"Ollama reachable; {want} available"
        return False, (
            f"Ollama is running but {want} is not pulled.\n"
            f"  Pull it with:  ollama pull {want}\n"
            f"  Installed: {', '.join(names) if names else '(none)'}")

    # ── one judgement ───────────────────────────────────────────────────
    def judge(self, img1: Path, img2: Path) -> dict[str, Any]:
        """Ask which of the two images shows the older person.

        Returns {older: 1|2|None, confidence, raw, latency_s, error}.
        `older` is relative to the order the images were passed in — the caller
        maps it back to left/right.
        """
        b64 = [encode_image(img1, self.cfg.max_image_px),
               encode_image(img2, self.cfg.max_image_px)]
        payload = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER, "images": b64},
            ],
            "stream": False,
            "format": SCHEMA,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                "num_ctx": self.cfg.num_ctx,
            },
        }

        last_err = ""
        for attempt in range(self.cfg.max_retries):
            t0 = time.time()
            try:
                r = self.session.post(self.url, json=payload, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                content = r.json().get("message", {}).get("content", "")
                older, conf = _parse(content)
                if older is None:
                    last_err = f"unparseable: {content[:120]}"
                    continue
                return {"older": older, "confidence": conf, "raw": content[:500],
                        "latency_s": round(time.time() - t0, 3), "error": None}
            except Exception as exc:                   # noqa: BLE001
                last_err = str(exc)
                time.sleep(1.5 * (attempt + 1))

        return {"older": None, "confidence": None, "raw": "",
                "latency_s": None, "error": last_err[:300]}


def _parse(text: str) -> tuple[int | None, float | None]:
    """Pull {older, confidence} out of the reply, tolerating stray prose."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            m2 = re.search(r"\b([12])\b", text)        # last-ditch: a bare 1 or 2
            return (int(m2.group(1)), None) if m2 else (None, None)
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, None
    if not isinstance(obj, dict):
        return None, None
    older = obj.get("older")
    try:
        older = int(older)
    except (TypeError, ValueError):
        return None, None
    if older not in (1, 2):
        return None, None
    conf = obj.get("confidence")
    try:
        conf = round(float(conf), 4)
    except (TypeError, ValueError):
        conf = None
    return older, conf


class MockJudge:
    """Offline stand-in so the pipeline can be exercised without a GPU.

    Judges by true age with injected noise and a deliberate left-side bias, so
    downstream analysis has something realistic to chew on.
    """

    def __init__(self, cfg: Config, ages: dict[str, int], accuracy: float = 0.78):
        import numpy as np
        self.cfg, self.ages, self.acc = cfg, ages, accuracy
        self.rng = np.random.default_rng(cfg.seed)

    def health(self) -> tuple[bool, str]:
        return True, "mock backend (no model required)"

    def judge(self, img1: Path, img2: Path) -> dict[str, Any]:
        a1 = self.ages.get(img1.name, 0)
        a2 = self.ages.get(img2.name, 0)
        truth = 1 if a1 > a2 else 2
        older = truth if self.rng.random() < self.acc else (3 - truth)
        if self.rng.random() < 0.06:                   # position bias
            older = 1
        return {"older": older, "confidence": round(float(self.rng.uniform(.5, 1)), 3),
                "raw": json.dumps({"older": older}), "latency_s": 0.01, "error": None}


class DeepSeekVisionJudge:
    """OpenAI-compatible vision API with credentials held only in memory."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.host != "https://api.deepseek.com":
            raise ValueError("DeepSeek Keychain credential requires https://api.deepseek.com")
        result = subprocess.run(
            ["security", "find-generic-password", "-s", cfg.keychain_service,
             "-a", cfg.keychain_account, "-w"], capture_output=True, text=True)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError("DeepSeek credential unavailable in macOS Keychain")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer " + result.stdout.strip()})

    def health(self) -> tuple[bool, str]:
        try:
            response = self.session.get(self.cfg.host + "/models", timeout=30)
            response.raise_for_status()
            models = response.json().get("data", [])
            model = next((m for m in models if m.get("id") == self.cfg.model_name), None)
            if not model or "image" not in model.get("input_modalities", []):
                return False, f"{self.cfg.model_name} not available with image input"
            return True, f"DeepSeek reachable; {self.cfg.model_name} supports images"
        except requests.RequestException:
            return False, "DeepSeek model check failed (network or authentication)"

    def judge(self, img1: Path, img2: Path) -> dict[str, Any]:
        content = [{"type": "text", "text": USER}]
        for image in (img1, img2):
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + encode_image(image, self.cfg.max_image_px)}})
        payload = {
            "model": self.cfg.model_name,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": self.cfg.temperature,
            "max_tokens": 256,
            "stream": False,
        }
        raw, error = "", ""
        start = time.monotonic()
        for attempt in range(self.cfg.max_retries):
            try:
                response = self.session.post(self.cfg.host + "/chat/completions",
                                             json=payload, timeout=self.cfg.timeout_s)
                if response.status_code in (400, 401, 402, 403, 404):
                    error = f"DeepSeek HTTP {response.status_code}"
                    break
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                older, confidence = _parse(raw)
                if older is None:
                    error = "DeepSeek response did not contain valid older choice"
                    continue
                return {"older": older, "confidence": confidence, "raw": raw[:500],
                        "latency_s": round(time.monotonic() - start, 3), "error": None,
                        "usage": body.get("usage", {}), "served_model": body.get("model")}
            except (requests.RequestException, ValueError, KeyError, IndexError):
                error = "DeepSeek request failed or returned invalid response"
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        return {"older": None, "confidence": None, "raw": raw[:500],
                "latency_s": round(time.monotonic() - start, 3), "error": error}


def make_judge(cfg: Config, mock: bool = False, ages: dict[str, int] | None = None):
    if mock:
        return MockJudge(cfg, ages or {})
    if cfg.provider == "deepseek":
        return DeepSeekVisionJudge(cfg)
    if cfg.provider != "ollama":
        raise ValueError(f"Unsupported provider: {cfg.provider}")
    return OllamaVisionJudge(cfg)
