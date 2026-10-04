from data_agent.llm.base import (
    LLMClient,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from data_agent.llm.scripted import ScriptedLLM

__all__ = ["LLMClient", "LLMResponse", "TokenUsage", "ToolCall", "ScriptedLLM", "build_client"]


def build_client(*, model: str | None = None) -> LLMClient:
    """按环境变量装配 LLM 客户端。

    没有配置密钥时直接抛错并说明怎么配，而不是悄悄退化成假模型——
    那样会让人以为跑出了真实结果。
    """
    import os

    from data_agent.llm.openai_client import OpenAICompatibleClient

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未配置模型密钥。请在 .env 或环境变量里设置 OPENAI_API_KEY"
            "（兼容 OpenAI 协议的服务都可以，用 OPENAI_BASE_URL 指向它），"
            "例如：export OPENAI_API_KEY=sk-xxx 与 "
            "export OPENAI_BASE_URL=https://api.deepseek.com/v1"
        )
    return OpenAICompatibleClient(model=model)
