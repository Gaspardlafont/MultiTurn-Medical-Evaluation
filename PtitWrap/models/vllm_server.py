"""Self-launching vLLM server backend — harness-managed, multi-model friendly.

The in-process ``vllm`` backend is perfect for ONE model, but two in-process
vLLM engines in the same process both grab GPU 0. To run a doctor model and a
separate judge model (see role separation), this backend instead **launches its
own vLLM OpenAI-compatible server subprocess**, waits until it's ready, talks to
it over HTTP (reusing the OpenAIChatLM client), and tears it down on exit.

So ``--model vllm-server`` and ``--judge_model vllm-server`` each spin up their
own server — same "harness launches it for you" UX as the single in-process
model, but for two models. Put them on one GPU with memory caps, or on separate
GPUs with ``cuda_visible_devices``.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import time
from typing import Any

from .base import register_model
from .openai_chat import OpenAIChatLM

logger = logging.getLogger(__name__)


@register_model("vllm-server")
class VLLMServerLM(OpenAIChatLM):
    """Launches a local vLLM server for a HF checkpoint, then chats over HTTP."""

    def __init__(
        self,
        pretrained: str,
        port: int = 8000,
        host: str = "127.0.0.1",
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 8192,
        dtype: str = "auto",
        cuda_visible_devices: str | int | None = None,
        enforce_eager: bool = False,
        startup_timeout: float = 600.0,
        log_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        base_url = f"http://{host}:{port}/v1"
        log_file = log_file or f"/tmp/vllm_server_{port}.log"

        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(pretrained),
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--dtype", str(dtype),
        ]
        # Opt-in: skip torch.compile / CUDA-graph capture. Some checkpoints
        # (e.g. the Apertus family with its xIELU activation falling back to a
        # Python impl) can otherwise spend 10-15 min compiling at load and race
        # the startup timeout. Slightly slower steady-state inference, but a
        # reliable, fast start. Off by default so existing configs are unchanged.
        if enforce_eager:
            cmd.append("--enforce-eager")
        logger.info("Launching vLLM server: %s (log: %s)", " ".join(cmd), log_file)
        self._log = open(log_file, "w")
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env
        )
        # Ensure the subprocess is killed even if the run crashes later.
        atexit.register(self.close)

        # Set up the HTTP client half (the served model name defaults to the
        # --model value, i.e. `pretrained`). The client connects lazily.
        super().__init__(
            model=str(pretrained), base_url=base_url, api_key="EMPTY", **kwargs
        )
        self._wait_until_ready(startup_timeout, log_file)

    def _wait_until_ready(self, timeout: float, log_file: str) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (rc={self.proc.returncode}). "
                    f"See {log_file}"
                )
            try:
                self.client.models.list()  # succeeds once the server is up
                logger.info("vLLM server ready at %s", self.client.base_url)
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError(
            f"vLLM server not ready within {timeout:.0f}s. See {log_file}"
        )

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()

    def __del__(self) -> None:
        # Best-effort cleanup; atexit is the primary guarantee.
        try:
            self.close()
        except Exception:
            pass
