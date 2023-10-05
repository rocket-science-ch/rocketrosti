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

from __future__ import annotations

import queue
import threading
import time as _unwrapped_time
from typing import Literal

import attrs
from loguru import logger
from typing_extensions import Self

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


class RateLimiter:
    """
    A tool to rate limit anything.

    Enforces a minimum time between operations.
    """

    @attrs.define
    class Params:
        initial_interval: float
        min_interval: float
        max_interval: float
        error_multiplier: float
        success_multiplier: float
        decrease_after: float

        @classmethod
        def from_config(cls) -> Self:
            return cls(
                initial_interval=config.openai_api.rate_limit_initial_interval,
                min_interval=config.openai_api.rate_limit_min_interval,
                max_interval=config.openai_api.rate_limit_max_interval,
                error_multiplier=config.openai_api.rate_limit_error_multiplier,
                success_multiplier=config.openai_api.rate_limit_success_multiplier,
                decrease_after=config.openai_api.rate_limit_decrease_after_seconds,
            )

        def __attrs_post_init__(self) -> None:
            assert self.min_interval <= self.initial_interval <= self.max_interval, (
                self.min_interval,
                self.initial_interval,
                self.max_interval,
            )
            assert self.min_interval > 0, self.min_interval
            assert self.error_multiplier >= 1, self.error_multiplier
            assert 0 < self.success_multiplier <= 1, self.success_multiplier
            assert self.min_interval <= self.max_interval, (self.min_interval, self.max_interval)

    class _Scope:
        _rate_limiter: RateLimiter
        _ok: bool

        def __init__(self, rate_limiter: RateLimiter) -> None:
            self._rate_limiter = rate_limiter
            self._ok = True

        def signal_error(self) -> None:
            assert self._ok, "Cannot signal error twice"
            self._ok = False
            self._rate_limiter._signal_rate_limit_hit()

        def __enter__(self) -> RateLimiter._Scope:
            return self

        def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
            if self._ok:
                self._rate_limiter._signal_success()

    _params: Params

    _ticket_queue: queue.Queue[None | Literal["stop"]]
    _started_queue: queue.Queue[None | Literal["stop"]]
    _result_queue: queue.Queue[bool | Literal["stop"]]

    # The thread that issues permissions to proceed
    _ticket_thread: threading.Thread

    def __init__(
        self,
        params: Params | None = None,
    ) -> None:
        if params is None:
            params = RateLimiter.Params.from_config()
        self._params = params
        self._result_queue = queue.Queue()
        self._started_queue = queue.Queue()
        self._ticket_queue = queue.Queue()

        self._ticket_thread = threading.Thread(target=self._ticket_thread_main, daemon=True)
        self._ticket_thread.start()

    def _ticket_thread_main(self) -> None:
        last_request = 0.0
        curr_interval = self._params.initial_interval

        while True:
            # Allow a call to proceed
            self._ticket_queue.put(None)
            # Wait for someone to take the ticket, or for decrease_after seconds to pass
            while True:
                try:
                    r = self._started_queue.get(timeout=self._params.decrease_after)
                    if r == "stop":
                        return
                except queue.Empty:
                    # timed out; gravitate towards initial interval
                    curr_interval = (3 * curr_interval + self._params.initial_interval) / 4
                    continue
                break

            # Now, we wait for self._curr_interval seconds, except if we get a result, in which
            # case we adjust the timeout.
            last_request = _SystemAdapter.time.time()
            wakeup_time = last_request + curr_interval
            sleep_seconds = wakeup_time - last_request
            while True:
                try:
                    result = self._result_queue.get(timeout=sleep_seconds if sleep_seconds > 0 else 0)
                    if result == "stop":
                        return
                except queue.Empty:
                    # timed out; allow next call to proceed
                    break
                if result:
                    curr_interval *= self._params.success_multiplier
                    curr_interval = max(curr_interval, self._params.min_interval)
                    logger.info("Rate limiter: success, new interval: {:g}", curr_interval)
                else:
                    curr_interval *= self._params.error_multiplier
                    curr_interval = min(curr_interval, self._params.max_interval)
                    logger.info("Rate limiter: error, new interval: {:g}", curr_interval)

    def _signal_rate_limit_hit(self) -> None:
        self._result_queue.put(False)

    def _signal_success(self) -> None:
        self._result_queue.put(True)

    def stop(self) -> None:
        """Stop the rate limiter."""
        self._started_queue.put("stop")
        self._result_queue.put("stop")

        # Empty the ticket queue, then put enough stops
        while True:
            try:
                self._ticket_queue.get_nowait()
            except queue.Empty:
                break
        for _ in range(10):
            # Each of these also requeue the stop, so one in theory is enough, but more can be
            # faster.
            self._ticket_queue.put("stop")

        self._ticket_thread.join()

    def __call__(self) -> _Scope:
        """Get a context manager that can be used to rate limit an operation."""
        # Wait for a ticket
        r = self._ticket_queue.get()
        if r == "stop":
            # Requeue; we don't know how many clients are waiting
            self._ticket_queue.put("stop")
            raise KeyboardInterrupt("Rate limiter stopped")
        self._started_queue.put(None)
        return self._Scope(self)
