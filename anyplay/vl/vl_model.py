"""Vision-language model: Qwen3.5-4B GGUF served with llama.cpp.

The model lives in the Hugging Face cache (``models--unsloth--Qwen3.5-4B-GGUF``):

    Qwen3.5-4B-Q4_K_M.gguf   main model (downloaded)
    mmproj-F16.gguf          vision projector (required for image input)

``llama serve`` (the unified llama.cpp binary, ``~/.local/bin/llama``) exposes
an OpenAI-compatible API on ``http://127.0.0.1:<port>/v1/chat/completions``.
Images go in as base64 data URLs inside the message content, exactly like the
OpenAI chat API.

The model is a hybrid thinking model: by default it emits reasoning tokens.
We pass ``chat_template_kwargs={"enable_thinking": ...}`` so callers get a
plain answer (default) or reasoning + answer (``thinking=True``).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_LLAMA_BIN = "~/.local/bin/llama"
HF_CACHE_REPO = Path.home() / ".cache/huggingface/hub" / "models--unsloth--Qwen3.5-4B-GGUF"


def find_hf_snapshot(repo: Path = HF_CACHE_REPO, pattern: str = "") -> Optional[Path]:
    """Find a file under ``snapshots/*/`` of a Hugging Face cache repo.

    ``pattern`` may include subdirectories, e.g. ``Qwen3.5-4B-Q4_K_M.gguf``.
    Returns the first match (sorted) or None.
    """
    matches = sorted(repo.glob(f"snapshots/*/{pattern}"))
    return matches[0] if matches else None


def default_model_path() -> Optional[Path]:
    return find_hf_snapshot(HF_CACHE_REPO, "Qwen3.5-4B-Q4_K_M.gguf")


def default_mmproj_path() -> Optional[Path]:
    return find_hf_snapshot(HF_CACHE_REPO, "mmproj-F16.gguf")


@dataclass
class VLConfig:
    model: str = ""        # main GGUF; "" = auto-detect in HF cache
    mmproj: str = ""       # vision projector GGUF; "" = auto-detect
    llama_bin: str = DEFAULT_LLAMA_BIN
    host: str = "127.0.0.1"
    port: int = 8198
    alias: str = "qwen3.5-4b"
    ctx_size: int = 8192
    gpu_layers: int = 999  # -ngl: 999 = offload everything the backend supports
    max_tokens: int = 1024
    temperature: float = 0.1
    thinking: bool = False  # default: plain answers, no reasoning tokens
    timeout: float = 300.0  # per-request HTTP timeout, seconds

    def resolved_model(self) -> Path:
        p = Path(self.model).expanduser() if self.model else default_model_path()
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"VL model GGUF not found (looked for Qwen3.5-4B-Q4_K_M.gguf in "
                f"{HF_CACHE_REPO}/snapshots/); download it or pass model=...")
        return p

    def resolved_mmproj(self) -> Path:
        p = Path(self.mmproj).expanduser() if self.mmproj else default_mmproj_path()
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"VL mmproj GGUF not found (looked for mmproj-F16.gguf in "
                f"{HF_CACHE_REPO}/snapshots/); download it or pass mmproj=...")
        return p

    def log_path(self) -> Path:
        return Path(f"/tmp/anyplay-vl-{self.port}.log")

    def pid_path(self) -> Path:
        return Path(f"/tmp/anyplay-vl-{self.port}.pid")


class VLModel:
    """Qwen3.5-4B behind llama.cpp's OpenAI-compatible server.

    The server is started on demand (``ensure_server``) and kept alive after a
    question so follow-up calls are fast. Stop it with ``stop_server`` (or
    ``python main.py vl-stop``).
    """

    def __init__(self, cfg: Optional[VLConfig] = None):
        self.cfg = cfg or VLConfig()
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------ server

    def _url(self, path: str) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}{path}"

    def _http(self, path: str, data: Optional[bytes] = None,
              timeout: Optional[float] = None) -> dict:
        req = urllib.request.Request(
            self._url(path), data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.cfg.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def is_running(self) -> bool:
        try:
            return self._http("/health", timeout=3).get("status") == "ok"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError):
            return False

    def start_server(self, wait: bool = True) -> subprocess.Popen:
        """Start ``llama serve`` in the background and wait until it is ready."""
        if self.is_running():
            return self._proc
        llama_bin = os.path.expanduser(self.cfg.llama_bin)
        llama = shutil.which(llama_bin) or llama_bin
        model, mmproj = self.cfg.resolved_model(), self.cfg.resolved_mmproj()
        cmd = [
            llama, "serve",
            "-m", str(model),
            "--mmproj", str(mmproj),
            "-ngl", str(self.cfg.gpu_layers),
            "-c", str(self.cfg.ctx_size),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "--alias", self.cfg.alias,
        ]
        log = self.cfg.log_path().open("ab", buffering=0)
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self.cfg.pid_path().write_text(str(self._proc.pid))
        if wait:
            self.wait_ready()
        return self._proc

    def wait_ready(self, timeout: float = 600.0) -> None:
        """Block until /health reports ok (model load + VRAM mapping)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return
            if self._proc is not None and self._proc.poll() is not None:
                log = self.cfg.log_path()
                tail = log.read_text(errors="replace")[-2000:] if log.exists() else ""
                raise RuntimeError(f"llama serve died (pid {self._proc.pid}); "
                                   f"log: {log}\n{tail}")
            time.sleep(2.0)
        raise TimeoutError(f"llama serve did not become ready in {timeout:.0f}s "
                           f"(log: {self.cfg.log_path()})")

    def stop_server(self) -> bool:
        """Stop a background server started by this module (pidfile-based)."""
        pidf = self.cfg.pid_path()
        if pidf.exists():
            try:
                pid = int(pidf.read_text().strip())
                os.kill(pid, 15)
                for _ in range(30):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pidf.unlink(missing_ok=True)
                        return True
                    time.sleep(0.5)
            except (ValueError, ProcessLookupError):
                pidf.unlink(missing_ok=True)
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        return not self.is_running()

    def ensure_server(self) -> None:
        if not self.is_running():
            self.start_server()

    # -------------------------------------------------------------------- ask

    @staticmethod
    def _image_part(path: Union[str, Path]) -> dict:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        mime = "image/png" if p.suffix.lower() in (".png",) else "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def ask(self, prompt: str, image: Optional[Union[str, Path]] = None,
            images: Optional[list] = None,
            system: Optional[str] = None,
            max_tokens: Optional[int] = None,
            thinking: Optional[bool] = None) -> dict:
        """Ask the model; returns ``{"content": str, "reasoning": str}``.

        ``image`` is a path to a screenshot (PNG/JPEG); ``images`` is a list
        of paths sent in order (oldest first) for temporal observations.
        ``thinking`` overrides the config: False = plain answer, True = also
        return reasoning.
        """
        self.ensure_server()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if not images and image is None:
            msgs.append({"role": "user", "content": prompt})
        else:
            parts = [{"type": "text", "text": prompt}]
            if images:
                parts += [self._image_part(p) for p in images]
            elif image is not None:
                parts.append(self._image_part(image))
            msgs.append({"role": "user", "content": parts})
        body = {
            "model": self.cfg.alias,
            "messages": msgs,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking":
                                     self.cfg.thinking if thinking is None
                                     else thinking},
        }
        j = self._http("/v1/chat/completions", data=json.dumps(body).encode())
        m = (j.get("choices") or [{}])[0].get("message") or {}
        return {"content": (m.get("content") or "").strip(),
                "reasoning": (m.get("reasoning_content") or "").strip()}

    def decide(self, obs, space, prompt: Optional[str] = None,
               max_tokens: int = 128) -> dict:
        """One decision step: temporal observation -> validated action.

        ``obs`` is a ``TemporalObservation`` (list of frames oldest first).
        Frames are written to temp PNGs for the OpenAI-compatible image API
        (the model server has no shared-memory channel; frames are small
        after capture-side scaling). The answer is parsed through
        ``parse_decision`` so only actions in ``space`` can be returned.

        Returns ``{"decision": Decision, "content": str, "ms": float}``.
        """
        import tempfile

        from ..actions.parser import parse_decision

        if prompt:
            text = prompt
        else:
            text = DECISION_PROMPT.format(vocabulary="\n".join(
                f"- {a.name}: {a.description}"
                for a in space.actions.values()))

        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="anyplay-obs-") as td:
            paths = []
            for i, frame in enumerate(obs.frames):
                p = Path(td) / f"t{i}.png"
                _write_png(p, frame)
                paths.append(p)
            r = self.ask(text, images=paths, max_tokens=max_tokens)
        decision = parse_decision(r["content"], space)
        return {"decision": decision, "content": r["content"],
                "ms": (time.time() - t0) * 1000.0}

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None


DECISION_PROMPT = """You are playing Shadow Dungeon, a side-view action game, as fast as you can.
You see the last three frames (oldest to newest) of the game view.

Choose exactly ONE action from this fixed list:
{vocabulary}

Rules:
- Reply with a single JSON object only, no markdown, no prose:
  {{"action": "<NAME>", "confidence": <0..1>, "reason": "<<=12 words>"}}
- Use the frame sequence to infer motion (enemies closing, player falling).
- If nothing needs doing, answer WAIT.
- Never invent actions that are not in the list."""


def _write_png(path: Path, frame) -> None:
    """Write an RGB uint8 frame as PNG without requiring cv2 (uses zlib)."""
    import struct
    import zlib

    arr = __import__("numpy").ascontiguousarray(frame)
    h, w = arr.shape[:2]
    raw = b"\x00" + arr.tobytes()  # filter byte 0 per row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(raw, 1)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", idat)
                     + chunk(b"IEND", b""))
