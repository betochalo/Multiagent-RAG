"""`_ask`: when the model spends max_tokens reasoning, retry with more room and charge the
tokens spent; never return an empty string as if it were an answer."""

import json

import pytest
from langchain_core.messages import AIMessage
from openai import LengthFinishReasonError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from investigation_agent.graph.nodes import LLMError, SolverNodes
from investigation_agent.services.trace import Tracer


class FlakyLLM:
    """Raises LengthFinishReasonError (what JSON mode does at the limit) `fails` times."""

    model_name = "flaky"

    def __init__(self, fails: int):
        self.fails, self.max_tokens_seen = fails, []

    def bind(self, **kwargs):
        self.max_tokens_seen.append(kwargs["max_tokens"])
        return self

    def invoke(self, messages):
        if len(self.max_tokens_seen) <= self.fails:
            completion = ChatCompletion(
                id="x", choices=[], created=0, model="flaky", object="chat.completion",
                usage=CompletionUsage(prompt_tokens=100, completion_tokens=self.max_tokens_seen[-1],
                                      total_tokens=100 + self.max_tokens_seen[-1]))
            raise LengthFinishReasonError(completion=completion)
        return AIMessage(content='{"approved": true}',
                         usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                         "total_tokens": 150},
                         response_metadata={"finish_reason": "stop"})


def _nodes(settings, llm, tmp_path):
    return SolverNodes(settings, llm, None, None, None, None, Tracer(tmp_path / "t.jsonl"),
                       tmp_path)


def test_length_error_is_retried_with_more_tokens(settings, tmp_path):
    llm = FlakyLLM(fails=1)
    nodes = _nodes(settings, llm, tmp_path)
    assert nodes._ask_json("critic", "s", "u") == {"approved": True}
    assert llm.max_tokens_seen == [settings.h200_max_tokens, settings.h200_max_tokens * 2]
    # The failed call is charged: its reasoning tokens were spent.
    assert nodes.tracer.tokens_out == settings.h200_max_tokens + 50
    calls = [json.loads(line) for line in open(tmp_path / "t.jsonl")]
    assert calls[0]["finish_reason"] == "length"


def test_length_error_at_the_cap_raises(settings, tmp_path):
    nodes = _nodes(settings, FlakyLLM(fails=99), tmp_path)
    with pytest.raises(LLMError, match="finish_reason=length"):
        nodes._ask_json("critic", "s", "u")
