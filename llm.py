import os
from collections.abc import Sequence

import httpx

import settings  # Load the project's .env for CLI use as well as Streamlit.


class LLMError(RuntimeError):
    """An unavailable or unusable DeepSeek response."""


def deepseek_enabled() -> bool:
    flag = os.getenv("DEEPSEEK_ENABLED", "1").strip().lower()
    return flag not in {"0", "false", "no"} and bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def deepseek_chat(messages: Sequence[dict[str, str]], *, json_mode: bool = False) -> str:
    if not deepseek_enabled():
        raise LLMError("DeepSeek 未启用。")

    api_key = os.environ["DEEPSEEK_API_KEY"].strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
    body = {
        "model": model,
        "messages": list(messages),
        "thinking": {"type": "disabled"},
        "max_tokens": 4096,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=35.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"DeepSeek 请求失败（HTTP {exc.response.status_code}）。") from None
    except httpx.TimeoutException:
        raise LLMError("DeepSeek 请求超时。") from None
    except (httpx.RequestError, httpx.InvalidURL):
        raise LLMError("暂时无法连接 DeepSeek。") from None

    try:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("DeepSeek 响应格式无效。") from None

    if finish_reason == "length":
        raise LLMError("DeepSeek 响应达到长度上限。")
    if not isinstance(content, str) or not content.strip():
        raise LLMError("DeepSeek 返回了空内容。")
    return content.strip()
