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

import threading
from typing import AsyncIterable, Iterable

import aioresponses
import openai
import pytest
from pytest_mock import MockerFixture

import rrosti.utils.config
from rrosti.llm_api import openai_api, openai_api_direct
from rrosti.utils.config import config

rrosti.utils.config.load_test_config()


@pytest.fixture(autouse=True)
def _no_network() -> Iterable[None]:
    with aioresponses.aioresponses():
        yield


FAKE_CHAT_COMPLETION: openai_api._ChatCompletionResponseDict = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-0613",
    "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
    "choices": [
        {"message": {"role": "assistant", "content": "\n\nThis is a test!"}, "finish_reason": "stop", "index": 0}
    ],
}


FAKE_EMBEDDING_RESPONSE: openai_api._EmbeddingResponseDict = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.0] * 512, "index": 0}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 8, "total_tokens": 8, "completion_tokens": 0},
}


@pytest.fixture
async def mock_chat_worker(mocker: MockerFixture) -> AsyncIterable[openai_api_direct._ChatCompletionWorker]:
    openai_api_direct._ChatCompletionWorker.delete_instance()

    # Patch the number of threads just to make backtraces nicer
    mocker.patch.object(config.openai_api, "num_chat_completion_worker_threads", 3)

    # No need to sleep so long in tests
    mocker.patch.object(config.openai_api, "rate_limit_initial_interval", 0.02)

    mocker.patch.object(openai_api_direct, "_create_chat_completion_internal", return_value=FAKE_CHAT_COMPLETION)
    yield openai_api_direct._ChatCompletionWorker.get_instance()
    openai_api_direct._ChatCompletionWorker.delete_instance()


@pytest.fixture
async def mock_embedding_worker(mocker: MockerFixture) -> AsyncIterable[openai_api_direct._EmbeddingWorker]:
    openai_api_direct._EmbeddingWorker.delete_instance()

    # Patch the number of threads just to make backtraces nicer
    mocker.patch.object(config.openai_api, "num_embedding_worker_threads", 3)

    # No need to sleep so long in tests
    mocker.patch.object(config.openai_api, "rate_limit_initial_interval", 0.02)
    mocker.patch.object(openai_api_direct, "_create_embedding_internal", return_value=FAKE_EMBEDDING_RESPONSE)
    yield openai_api_direct._EmbeddingWorker.get_instance()
    openai_api_direct._EmbeddingWorker.delete_instance()


@pytest.fixture
async def mock_provider(
    mock_chat_worker: openai_api_direct._ChatCompletionWorker, mock_embedding_worker: openai_api_direct._EmbeddingWorker
) -> openai_api_direct.DirectOpenAIApiProvider:
    return openai_api_direct.DirectOpenAIApiProvider()


async def test_chat_completion_worker_async(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    result = await mock_provider.acreate_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    assert result is FAKE_CHAT_COMPLETION


async def test_chat_completion_worker_sync(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    # janus.Queue needs an event loop, hence this is async

    result = mock_provider.create_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    assert result is FAKE_CHAT_COMPLETION


async def test_embedding_worker_async(
    mock_provider: openai_api_direct.DirectOpenAIApiProvider,
) -> None:
    result = await mock_provider.acreate_embedding(input=["Hello, world!"])

    assert result.embeddings.shape == (1, 512)


async def test_embedding_worker_sync(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    result = mock_provider.create_embedding(input=["Hello, world!"])

    assert result.embeddings.shape == (1, 512)


class MyTestException(Exception):
    pass


async def test_chat_completion_worker_exception_async(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = MyTestException()  # type: ignore[attr-defined]
    with pytest.raises(MyTestException):
        await mock_provider.acreate_chat_completion(
            model="test_model",
            messages=[{"role": "system", "content": "Hello, world!"}],
        )


async def test_chat_completion_worker_exception_sync(
    mock_provider: openai_api_direct.DirectOpenAIApiProvider,
) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = MyTestException()  # type: ignore[attr-defined]
    with pytest.raises(MyTestException):
        mock_provider.create_chat_completion(
            model="test_model",
            messages=[{"role": "system", "content": "Hello, world!"}],
        )


class MockTime:
    _now: float

    def __init__(self) -> None:
        self._now = 1000.0

    def time(self) -> float:
        return self._now

    def sleep(self, duration: float) -> None:
        self._now += duration

    async def sleep_async(self, duration: float) -> None:
        self._now += duration

    def wait_cond_with_timeout(self, cond: threading.Condition, timeout: float) -> bool:
        timed_out = cond.wait(timeout=timeout)
        if timed_out:
            self._now += timeout
        return timed_out


@pytest.fixture
def _mock_time(mocker: MockerFixture) -> None:
    _mock_time = MockTime()
    mocker.patch.object(openai_api_direct._SystemAdapter.time, "time", wraps=_mock_time.time)


@pytest.mark.timeout(3)
@pytest.mark.usefixtures("_mock_time")
async def test_ratelimit_error_async(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = [  # type: ignore[attr-defined]
        openai.error.RateLimitError(),  # type: ignore[no-untyped-call]
        FAKE_CHAT_COMPLETION,
    ]
    await mock_provider.acreate_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    # TODO: a proper test
    # openai_api._SystemAdapter.time.sleep.assert_called_once_with(  # type: ignore[attr-defined]
    #     openai_api.params.RATE_LIMIT_INITIAL_INTERVAL * openai_api.params.RATE_LIMIT_ERROR_MULTIPLIER
    # )


@pytest.mark.timeout(3)
@pytest.mark.usefixtures("_mock_time")
async def test_ratelimit_error_sync(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = [  # type: ignore[attr-defined]
        openai.error.RateLimitError(),  # type: ignore[no-untyped-call]
        FAKE_CHAT_COMPLETION,
    ]
    mock_provider.create_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    # TODO: a proper test
    # openai_api._SystemAdapter.time.sleep.assert_called_once_with(  # type: ignore[attr-defined]
    #     openai_api.params.RATE_LIMIT_INITIAL_INTERVAL * openai_api.params.RATE_LIMIT_ERROR_MULTIPLIER
    # )


@pytest.mark.timeout(3)
@pytest.mark.usefixtures("_mock_time")
async def test_ratelimit_success_async(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = [  # type: ignore[attr-defined]
        FAKE_CHAT_COMPLETION,
        FAKE_CHAT_COMPLETION,
    ]
    await mock_provider.acreate_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )
    await mock_provider.acreate_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    # TODO: a proper test
    # openai_api._SystemAdapter.time.sleep.assert_called_once_with(  # type: ignore[attr-defined]
    #     openai_api.params.RATE_LIMIT_INITIAL_INTERVAL * openai_api.params.RATE_LIMIT_SUCCESS_MULTIPLIER
    # )


@pytest.mark.timeout(3)
@pytest.mark.usefixtures("_mock_time")
async def test_ratelimit_success_sync(mock_provider: openai_api_direct.DirectOpenAIApiProvider) -> None:
    openai_api_direct._create_chat_completion_internal.side_effect = [  # type: ignore[attr-defined]
        FAKE_CHAT_COMPLETION,
        FAKE_CHAT_COMPLETION,
    ]
    mock_provider.create_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )
    mock_provider.create_chat_completion(
        model="test_model",
        messages=[{"role": "system", "content": "Hello, world!"}],
    )

    # TODO: a proper test
    # openai_api._SystemAdapter.time.sleep.assert_called_once_with(  # type: ignore[attr-defined]
    #     openai_api.params.RATE_LIMIT_INITIAL_INTERVAL * openai_api.params.RATE_LIMIT_SUCCESS_MULTIPLIER
    # )
