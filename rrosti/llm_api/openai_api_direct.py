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

"""An implementation of OpenAIApiProvider that directly calls the OpenAI API."""

from __future__ import annotations

import asyncio
import concurrent
import threading
import time as _unwrapped_time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

import attrs
import janus
import openai
from loguru import logger
from overrides import override
from typing_extensions import Self

from rrosti.llm_api import openai_api as api
from rrosti.llm_api.rate_limiter import RateLimiter
from rrosti.query import logging as qlog
from rrosti.utils.config import config


class _SystemAdapter:
    """
    This static class wraps some system operations that we need to mock in tests without influencing the rest of the
    system.
    """

    class time:
        @staticmethod
        def time() -> float:
            return _unwrapped_time.time()

    class asyncio:
        @staticmethod
        async def sleep(seconds: float) -> None:
            await asyncio.sleep(seconds)


@attrs.frozen
class ChatCompletionRequest:
    model: str
    messages: list[api._ChatCompletionMessageDict]
    max_tokens: int | None
    temperature: float


@attrs.frozen
class EmbeddingRequest:
    model: str
    input: list[str]


class DirectOpenAIApiProvider(api.OpenAIApiProvider):
    """An OpenAIApiProvider that directly calls the OpenAI API."""

    _chat_worker: _ChatCompletionWorker
    _embedding_worker: _EmbeddingWorker

    def __init__(self) -> None:
        self._chat_worker = _ChatCompletionWorker.get_instance()
        self._embedding_worker = _EmbeddingWorker.get_instance()

        endpoint = config.openai_api.endpoint
        openai.api_key = endpoint.api_key
        if endpoint.api_type is not None:
            openai.api_type = endpoint.api_type
        if endpoint.api_version is not None:
            openai.api_version = endpoint.api_version
        if endpoint.api_base is not None:
            openai.api_base = endpoint.api_base

    @override
    def _create_chat_completion_raw(
        self,
        *,
        model: str,
        messages: list[api._ChatCompletionMessageDict],
        max_tokens: int,
        temperature: float,
    ) -> api._ChatCompletionResponseDict:
        return self._chat_worker.work(
            ChatCompletionRequest(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        )

    @override
    async def _acreate_chat_completion_raw(
        self,
        *,
        model: str,
        messages: list[api._ChatCompletionMessageDict],
        max_tokens: int,
        temperature: float,
    ) -> api._ChatCompletionResponseDict:
        return await self._chat_worker.awork(
            ChatCompletionRequest(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        )

    @override
    def _create_embedding_raw(
        self,
        input: list[str],
        *,
        model: str,
    ) -> api._EmbeddingResponseDict:
        return self._embedding_worker.work(EmbeddingRequest(model=model, input=input))

    @override
    async def _acreate_embedding_raw(
        self,
        input: list[str],
        *,
        model: str,
    ) -> api._EmbeddingResponseDict:
        return await self._embedding_worker.awork(EmbeddingRequest(model=model, input=input))


def _map_model(model: str) -> dict[str, str]:
    engine_map = config.openai_api.endpoint.engine_map
    if engine_map is not None:
        return dict(engine=engine_map[model])
    return dict(model=model)


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class WorkItem(Generic[RequestT, ResponseT]):
    """
    A work item that can be used synchronously or asynchronously.

    For example, a request to the embedding API or the chat completion API.
    """

    request: RequestT
    fut: _SyncFuture[ResponseT] | _AsyncFuture[ResponseT]
    time_created: float
    time_dequeued: float | None
    time_started: float | None
    time_completed: float | None

    _private_object = object()

    def __init__(self, private_object: object) -> None:
        """Private constructor. Do not call."""
        assert private_object is self._private_object, "Do not call this constructor"

    def _initialize(
        self,
        request: RequestT,
        fut: concurrent.futures.Future[ResponseT] | asyncio.Future[ResponseT],
    ) -> None:
        self.request = request
        self.fut = fut
        self.time_created = _SystemAdapter.time.time()
        self.time_dequeued = None
        self.time_completed = None

    @classmethod
    def make_sync(cls, request: RequestT) -> WorkItem[RequestT, ResponseT]:
        """Make a synchronous work item."""
        item = cls(cls._private_object)
        item._initialize(request, _SyncFuture())
        return item

    @classmethod
    def make_async(cls, request: RequestT) -> WorkItem[RequestT, ResponseT]:
        """Make an asynchronous work item."""
        item = cls(cls._private_object)
        item._initialize(request, _AsyncFuture())
        return item

    def set_result(self, result: ResponseT) -> None:
        """Set the result of the future."""
        self.time_completed = _SystemAdapter.time.time()
        if isinstance(self.fut, _SyncFuture):
            self.fut.set_result(result)
        else:
            assert isinstance(self.fut, _AsyncFuture)
            self.fut.get_loop().call_soon_threadsafe(self.fut.set_result, result)

    def set_exception(self, exc: Exception) -> None:
        """Set the exception of the future."""
        self.time_completed = _SystemAdapter.time.time()
        if isinstance(self.fut, _SyncFuture):
            self.fut.set_exception(exc)
        else:
            assert isinstance(self.fut, _AsyncFuture)
            self.fut.get_loop().call_soon_threadsafe(self.fut.set_exception, exc)


_RequestWorkerT = TypeVar("_RequestWorkerT", bound="RequestWorker[Any, Any]")


class RequestWorker(ABC, Generic[RequestT, ResponseT]):
    """A worker with a queue of requests that can be used synchronously or asynchronously."""

    _instances: ClassVar[dict[type, Any]] = {}

    _queue: janus.Queue[WorkItem[RequestT, ResponseT] | Literal["stop"]]
    _threads: list[threading.Thread]
    _rate_limiter: RateLimiter

    @classmethod
    def get_instance(cls) -> Self:
        """Get an instance of the given class, creating it if necessary."""
        if cls is RequestWorker:
            raise TypeError("Cannot instantiate abstract class RequestWorker")
        if cls not in RequestWorker._instances:
            # This requires cls to have a no-argument constructor
            RequestWorker._instances[cls] = cls()  # type: ignore[call-arg]
        return cast(Self, RequestWorker._instances[cls])

    def _finalize_on_delete(self) -> None:
        pass

    @classmethod
    def delete_instance(cls: type[_RequestWorkerT]) -> None:
        """Delete the instance of the given class, if it exists."""
        if cls in RequestWorker._instances:
            RequestWorker._instances[cls]._finalize_on_delete()
            del RequestWorker._instances[cls]

    class RateLimitException(Exception):
        pass

    def stop(self) -> None:
        """Stop the worker. Useful for testing and orderly shutdown."""
        logger.info("Stopping {}...", type(self).__name__)
        for _ in self._threads:
            self._queue.sync_q.put("stop")
        for thread in self._threads:
            thread.join()
        self._rate_limiter.stop()
        logger.info("Stopped {}.", type(self).__name__)

    def __init__(self, num_threads: int) -> None:
        if type(self) is RequestWorker:
            raise TypeError("Cannot instantiate abstract class RequestWorker")

        logger.info("Initializing {} with {} threads", type(self).__name__, num_threads)
        self._queue = janus.Queue()
        self._threads = [
            threading.Thread(target=self._thread_main, daemon=True, name="{i}-worker-{type(self)}")
            for i in range(num_threads)
        ]
        self._rate_limiter = RateLimiter()
        for thread in self._threads:
            thread.start()

    def _check_request(self, req: RequestT) -> None:
        pass

    def work(self, request: RequestT) -> ResponseT:
        """Work on a request synchronously."""
        self._check_request(request)
        item = WorkItem[RequestT, ResponseT].make_sync(request)
        self._queue.sync_q.put(item)
        return item.fut.result()

    async def awork(self, request: RequestT) -> ResponseT:
        """Work on a request asynchronously."""
        self._check_request(request)
        item = WorkItem[RequestT, ResponseT].make_async(request)
        await self._queue.async_q.put(item)
        await cast(Any, item.fut)
        return item.fut.result()

    @abstractmethod
    def _process_one(self, request: RequestT) -> ResponseT: ...

    def _thread_main(self) -> None:
        # logger.info("Thread {} starting", threading.current_thread().name)
        while True:
            item = self._queue.sync_q.get()
            if item == "stop":
                return
            item.time_dequeued = _SystemAdapter.time.time()
            while True:  # loop until we get a response
                with self._rate_limiter() as scope:
                    item.time_started = _SystemAdapter.time.time()
                    try:
                        response = self._process_one(item.request)
                    except RequestWorker.RateLimitException:
                        logger.info("{}: Rate limit reached.", str(type(self).__name__))
                        scope.signal_error()
                        continue
                    except Exception as e:
                        item.time_completed = _SystemAdapter.time.time()
                        item.set_exception(e)
                    else:
                        item.time_completed = _SystemAdapter.time.time()
                        item.set_result(response)
                    break


class _ChatCompletionWorker(RequestWorker[ChatCompletionRequest, api._ChatCompletionResponseDict]):
    """A worker with a queue of chat completion requests that can be used synchronously or asynchronously."""

    def __init__(self) -> None:
        super().__init__(num_threads=config.openai_api.num_chat_completion_worker_threads)

    @override
    def _check_request(self, req: ChatCompletionRequest) -> None:
        if not req.messages:
            raise ValueError("Must have at least one message")

    @override
    def _process_one(self, request: ChatCompletionRequest) -> api._ChatCompletionResponseDict:
        logger.info("Getting chat completion for {} messages", len(request.messages))
        try:
            start_time = _SystemAdapter.time.time()
            resp = _create_chat_completion_internal(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except openai.error.RateLimitError as e:
            raise RequestWorker.RateLimitException from e
        except Exception as e:
            logger.error("Error during chat completion: {}", e)
            raise

        end_time = _SystemAdapter.time.time()
        elapsed_time = end_time - start_time
        logger.info("Got chat completion in {:.3f} seconds.", elapsed_time)
        return resp


class _EmbeddingWorker(RequestWorker[EmbeddingRequest, api._EmbeddingResponseDict]):
    """A worker with a queue of embedding requests that can be used synchronously or asynchronously."""

    def __init__(self) -> None:
        super().__init__(num_threads=config.openai_api.num_embedding_worker_threads)

    @override
    def _check_request(self, req: EmbeddingRequest) -> None:
        if not req.input:
            raise ValueError("Must have at least one prompt")

    @override
    def _process_one(self, request: EmbeddingRequest) -> api._EmbeddingResponseDict:
        logger.info("Getting embedding for {} prompts", len(request.input))
        while True:
            try:
                start_time = _SystemAdapter.time.time()
                resp = _create_embedding_internal(
                    model=request.model,
                    input=request.input,
                )
            except openai.error.RateLimitError as e:
                raise RequestWorker.RateLimitException from e
            except Exception as e:
                logger.error("Error during embedding: {}", e)
                raise

            end_time = _SystemAdapter.time.time()
            elapsed_time = end_time - start_time
            logger.info("Got {} embeddings in {:.3f} seconds.", len(request.input), elapsed_time)
            return resp


def _create_chat_completion_internal(
    *, model: str, messages: list[api._ChatCompletionMessageDict], max_tokens: int | None, temperature: float
) -> api._ChatCompletionResponseDict:
    assert messages, "Must have at least one message"

    with qlog.ChatCompletionCallEvent.section(model, messages, openai.api_base):
        return cast(
            api._ChatCompletionResponseDict,
            openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **_map_model(model),
            ),
        )


_SyncFuture = concurrent.futures.Future
_AsyncFuture = asyncio.Future


def _create_embedding_internal(
    *,
    model: str,
    input: list[str],
) -> api._EmbeddingResponseDict:
    """
    Create an embedding synchronously.

    Wraps openai.Embedding.create, handling Azure etc. transparently.
    """
    assert input, "Must have at least one prompt"

    return cast(
        api._EmbeddingResponseDict,
        openai.Embedding.create(input=input, **_map_model(model)),  # type: ignore[no-untyped-call]
    )
