"""LangChain chat model on the H200 vLLM, for the LangGraph nodes that call an LLM.

vLLM speaks the OpenAI API, so the model is built with the `openai` provider pointed at
the H200. The model id is not configured: it is read from `/v1/models` through the H200
client, because the lab changed the served model at least once this semester.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from investigation_agent.config.h200 import get_h200_client
from investigation_agent.config.settings import Settings


def get_chat_model(settings: Settings, port: int | None = None) -> BaseChatModel:
    """Build the LangChain chat model for the H200 from the given settings. `port` selects
    a replica (the lab serves the same model on h200_port and h200_replica_port)."""
    model_id = get_h200_client(settings).model
    return init_chat_model(
        f"openai:{model_id}",
        base_url=f"http://{settings.h200_host}:{port or settings.h200_port}/v1",
        # vLLM does not validate the key; "local" is not a credential
        api_key="local",
        temperature=settings.h200_temperature,
        max_tokens=settings.h200_max_tokens,
        timeout=settings.h200_timeout,
        # The model always reasons; this only asks vLLM to split the reasoning out of
        # `content`. Reasoning is billed from the same max_tokens budget.
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
