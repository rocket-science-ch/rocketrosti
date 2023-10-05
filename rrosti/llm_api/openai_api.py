# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Interfaces to query the OpenAI API.

The abstract interface OpenAIApiProvider has implementations that either directly call the OpenAI API
(OpenAIApiProvider) or proxy through a centralized server (OpenAIApiProxyProvider).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict, TypeVar

import numpy as np

from rrosti.utils.config import FromConfig, ModelCost, config, uses_config
from rrosti.utils.misc import FloatArray

T = TypeVar("T")


# Here's an example of an API response:
# {
#   "data": [
#     {
#       "embedding": [
#         -0.00239578727632761,
#         # ...
#         -0.02301216684281826
#       ],
#       "index": 0,
#       "object": "embedding"
#     }
#   ],
#   "model": "text-embedding-ada-002-v2",
#   "object": "list",
#   "usage": {
#     "prompt_tokens": 6,
#     "total_tokens": 6
#   }
# }


def format_cost(cost: float) -> str:
    # if cost < 0.01 and cost != 0:
    #     return f"{cost * 100:.5f}".rstrip("0").rstrip(".") + " ¢ (USD)"

    return f"{cost:.8f}".rstrip("0").rstrip(".") + " USD"


def _get_model_cost_per_1k_tokens(model: str) -> ModelCost:
    try:
        return config.openai_api.model_cost[model]
    except KeyError:
        for k, v in config.openai_api.model_cost.items():
            if model.startswith(k):
                return v
    raise ValueError(f"Unknown model: {model}")


class _UsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int  # only present for completions
    total_tokens: int


class _ChatCompletionMessageDict(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class _ChatCompletionResponseItemDict(TypedDict):
    index: int
    message: _ChatCompletionMessageDict
    finish_reason: Literal["stop", "length"]


class _ChatCompletionResponseDict(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[_ChatCompletionResponseItemDict]
    usage: _UsageDict


class _EmbeddingResponseItem(TypedDict):
    object: Literal["embedding"]
    embedding: list[float]
    index: int


class _EmbeddingResponseDict(TypedDict):
    object: Literal["list"]
    data: list[_EmbeddingResponseItem]
    model: str
    usage: _UsageDict


@dataclass(frozen=True)
class EmbeddingResponse:
    snippets: list[str]  # (N,)
    embeddings: FloatArray  # (N, emb_size)
    model: str
    prompt_tokens: int

    @property
    def cost(self) -> float:
        return _get_model_cost_per_1k_tokens(self.model).calculate(self.prompt_tokens, 0)

    @property
    def cost_str(self) -> str:
        return format_cost(self.cost)

    @staticmethod
    def from_dict(snippets: list[str], raw: _EmbeddingResponseDict) -> EmbeddingResponse:
        model = raw["model"]
        prompt_tokens = raw["usage"]["prompt_tokens"]
        # Verify that the data has the right number of elements
        assert len(raw["data"]) == len(snippets), raw
        # The API docs warn us that the order may be different, so sort by the index field
        raw["data"].sort(key=lambda d: d["index"])
        assert [d["index"] for d in raw["data"]] == list(range(len(snippets))), raw
        embeddings = np.array([raw["data"][i]["embedding"] for i in range(len(snippets))], dtype=np.float32)
        return EmbeddingResponse(
            snippets=snippets,
            embeddings=embeddings,
            model=model,
            prompt_tokens=prompt_tokens,
        )

    def __repr__(self) -> str:
        return (
            f"EmbeddingResponse(snippets={self.snippets!r}, model={self.model!r}, "
            f"prompt_tokens={self.prompt_tokens!r}), cost={format_cost(self.cost)}"
        )


class OpenAIApiProvider(ABC):
    """An abstract class that provides the OpenAI API."""

    @abstractmethod
    def _create_chat_completion_raw(
        self,
        *,
        model: str,
        messages: list[_ChatCompletionMessageDict],
        max_tokens: int,
        temperature: float,
    ) -> _ChatCompletionResponseDict:
        """
        Create a chat completion synchronously.

        Wraps openai.ChatCompletion.create, handling Azure etc. transparently.

        This is the raw method that subclasses should override.
        """

    @uses_config
    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[_ChatCompletionMessageDict],
        max_tokens: int = FromConfig("openai_api.completion_max_tokens"),
        temperature: float = FromConfig("openai_api.completion_temperature"),
    ) -> _ChatCompletionResponseDict:
        """
        Create a chat completion synchronously.

        Wraps openai.ChatCompletion.create, handling Azure etc. transparently.
        """

        assert messages, "Must have at least one message"

        return self._create_chat_completion_raw(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @abstractmethod
    async def _acreate_chat_completion_raw(
        self,
        *,
        model: str,
        messages: list[_ChatCompletionMessageDict],
        max_tokens: int,
        temperature: float,
    ) -> _ChatCompletionResponseDict:
        """
        Create a chat completion asynchronously.

        Wraps openai.ChatCompletion.create, handling Azure etc. transparently.

        This is the raw method that subclasses should override.
        """

    @uses_config
    async def acreate_chat_completion(
        self,
        *,
        model: str,
        messages: list[_ChatCompletionMessageDict],
        max_tokens: int = FromConfig("openai_api.completion_max_tokens"),
        temperature: float = FromConfig("openai_api.completion_temperature"),
    ) -> _ChatCompletionResponseDict:
        """
        Create a chat completion asynchronously.

        Wraps openai.ChatCompletion.create, handling Azure etc. transparently.
        """

        assert messages, "Must have at least one message"

        if max_tokens is None:
            max_tokens = config.openai_api.completion_max_tokens

        if temperature is None:
            temperature = config.openai_api.completion_temperature

        return await self._acreate_chat_completion_raw(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @abstractmethod
    async def _acreate_embedding_raw(
        self,
        input: list[str],
        *,
        model: str,
    ) -> _EmbeddingResponseDict:
        """
        Create an embedding asynchronously.

        Wraps openai.Embedding.create, handling Azure etc. transparently.

        Returns an _EmbeddingResponseDict, which is a TypedDict of what is returned by the API.
        """

    @abstractmethod
    def _create_embedding_raw(
        self,
        input: list[str],
        *,
        model: str,
    ) -> _EmbeddingResponseDict:
        """
        Create an embedding synchronously.

        Wraps openai.Embedding.create, handling Azure etc. transparently.

        Returns an _EmbeddingResponseDict, which is a TypedDict of what is returned by the API.
        """

    async def acreate_embedding(
        self,
        input: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Create an embedding asynchronously.

        Wraps openai.Embedding.create, handling Azure etc. transparently.
        """

        assert input, "Must have at least one prompt"

        if model is None:
            model = config.openai_api.embedding_model

        return EmbeddingResponse.from_dict(input, await self._acreate_embedding_raw(input, model=model))

    def create_embedding(
        self,
        input: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Create an embedding synchronously.

        Wraps openai.Embedding.create, handling Azure etc. transparently.
        """

        assert input, "Must have at least one prompt"

        if model is None:
            model = config.openai_api.embedding_model

        return EmbeddingResponse.from_dict(input, self._create_embedding_raw(input, model=model))
