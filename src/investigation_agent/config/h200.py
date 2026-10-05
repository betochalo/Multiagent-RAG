"""Connection to the USFQ H200: LLM on vLLM and embeddings on Ollama. Standard library only.

Based on `solver-v2/h200.py` from the Taller 03 v2 kit.

- **No model id is hardcoded.** It is read from `/v1/models` on first use, so building the
  client does not need the VPN; the first call does.
- **Reasoning cannot be turned off.** `enable_thinking: True` only asks vLLM to SPLIT it
  into `reasoning`; the model still reasons and bills it as output tokens. If it runs out
  of `max_tokens` while reasoning, `content` comes back empty with `finish_reason=length`:
  `chat` detects it and retries with a larger budget instead of returning an empty string.
"""

import json
import time
import urllib.error
import urllib.request
from functools import cached_property

from investigation_agent.config.settings import Settings


class ReasoningExhaustedError(RuntimeError):
    """The model spent all of `max_tokens` reasoning and wrote no `content`."""


class H200Client:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def model(self) -> str:
        return self._request(self.settings.h200_port, "v1/models", timeout=8)["data"][0]["id"]

    def _request(self, port: int, path: str, body: dict | None = None,
                 timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            f"http://{self.settings.h200_host}:{port}/{path}",
            data=None if body is None else json.dumps(body).encode(),
            # vLLM does not validate the key; "local" is not a credential
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.settings.h200_timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"H200 returned {err.code}: {err.read()[:300]!r}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"No response from {self.settings.h200_host}:{port} "
                               f"({err.reason}). Is GlobalProtect connected?") from err

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, max_tokens: int | None = None,
             temperature: float | None = None) -> dict:
        """One logical call. Returns the message, the separated reasoning, the usage summed
        over all attempts, the latency, and how many attempts it took."""
        max_tokens = max_tokens or self.settings.h200_max_tokens
        temperature = self.settings.h200_temperature if temperature is None else temperature
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        latency, attempts = 0.0, 0
        while True:
            attempts += 1
            r = self._single_call(messages, tools, json_mode, max_tokens, temperature)
            for k in usage:
                usage[k] += r["usage"].get(k, 0)
            latency += r["latency_s"]
            empty = not r["content"].strip() and not r["tool_calls"]
            if not (r["finish_reason"] == "length" and empty):
                return {**r, "usage": usage, "latency_s": round(latency, 2), "attempts": attempts}
            if max_tokens >= self.settings.h200_max_tokens_cap:
                raise ReasoningExhaustedError(
                    f"empty content with finish_reason=length after {attempts} attempt(s); "
                    f"last max_tokens={max_tokens}, {len(r['reasoning'])} chars of reasoning")
            max_tokens = min(max_tokens * 2, self.settings.h200_max_tokens_cap)

    def _single_call(self, messages: list[dict], tools: list[dict] | None, json_mode: bool,
                     max_tokens: int, temperature: float) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": True}}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = self._request(self.settings.h200_port, "v1/chat/completions", body)
        choice = r["choices"][0]
        m = choice["message"]
        return {
            "content": m.get("content") or "",
            "tool_calls": m.get("tool_calls") or [],
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": r.get("usage", {}),
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Normalized embeddings (dot product is cosine), in batches."""
        batch = self.settings.h200_embed_batch
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            r = self._request(self.settings.h200_embed_port, "v1/embeddings",
                              {"model": self.settings.h200_embed_model,
                               "input": texts[i:i + batch]})
            for d in sorted(r["data"], key=lambda d: d["index"]):
                v = d["embedding"]
                n = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / n for x in v])
        return out


def get_h200_client(settings: Settings) -> H200Client:
    """Build an H200 client from the given settings."""
    return H200Client(settings)
