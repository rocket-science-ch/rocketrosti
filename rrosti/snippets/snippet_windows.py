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


"""Tools to split a text into overlapping windows of a given size."""

import re
from dataclasses import dataclass
from typing import Iterable

from rrosti.snippets._page import Page


@dataclass
class WindowedSnippetInfo:
    text: str
    start: int
    end: int

    # These are 0 for windows()
    page_start: int
    page_end: int


# TODO: This is apparently a bit slow.
def windows(text: str, window_size: int, step_size: int) -> Iterable[WindowedSnippetInfo]:
    """Split a text snippet into windows of a given size, with a given stride."""
    BOUNDARY_RE = re.compile(r"\b", re.DOTALL)
    start = 0
    prev_end = -1
    while start < len(text):
        # Start and end at word boundary. Find the largest window that fits.
        m = BOUNDARY_RE.search(text, start)
        if m is None:
            break
        start = m.start()
        # eat whitespace
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        assert start == 0 or not (text[start - 1].isspace() and text[start].isspace()), text[start - 1 : start + 1]

        end = None
        for match in BOUNDARY_RE.finditer(text, start):
            if match.start() - start > window_size:
                break
            end = match.start()
        if end is None:
            break

        assert end - start <= window_size, (start, end, window_size)
        assert start == 0 or not (text[start - 1].isspace() and text[start].isspace()), text[start - 1 : start + 1]

        if end > prev_end:
            yield WindowedSnippetInfo(text=text[start:end], start=start, end=end, page_start=0, page_end=0)
        prev_end = end

        start += step_size


def windows_with_source(
    pages: list[Page],
    join_str: str,
    window_size: int,
    step_size: int,
) -> Iterable[WindowedSnippetInfo]:
    """
    Split a text snippet into windows of a given size, with a given stride.

    Like windows(), but takes pages separately and fills the page_start and page_end fields.
    """

    # Construct the text body and the indices of the page boundaries
    page_boundaries = [0]
    texts = []
    page_numbers = [1]
    for page in pages:
        page_boundaries.append(page_boundaries[-1] + len(page.text) + len(join_str))
        texts.append(page.text)
        page_numbers.append(page.page_number)
    text = join_str.join(texts)

    for window in windows(text, window_size, step_size):
        # Find the page boundaries that this window overlaps
        page_start = None
        page_end = None
        for i, boundary in enumerate(page_boundaries):
            if boundary > window.start:
                page_start = page_numbers[i]
                break
        for i, boundary in enumerate(page_boundaries):
            if boundary > window.end:
                page_end = page_numbers[i]
                break

        assert page_start is not None
        assert page_end is not None

        yield WindowedSnippetInfo(
            text=window.text,
            start=window.start,
            end=window.end,
            page_start=page_start,
            page_end=page_end,
        )
