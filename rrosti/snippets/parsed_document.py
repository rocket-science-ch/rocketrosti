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

import hashlib
from pathlib import Path
from typing import Any

import attrs

from rrosti.snippets import snippet_windows
from rrosti.snippets._page import Page
from rrosti.snippets.snippet import Snippet
from rrosti.utils.config import config


@attrs.frozen
class ParsedDocument:
    """
    A parsed document contains our computed data about a document (such as a PDF file).

    Identifies the document by name and sha256 checksum. Contains the extracted snippets.
    """

    name: str | None = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(str)))
    path: str = attrs.field(validator=attrs.validators.instance_of(str))
    sha256: str = attrs.field(validator=attrs.validators.instance_of(str))
    pages: list[Page] = attrs.field(
        validator=attrs.validators.deep_iterable(member_validator=attrs.validators.instance_of(Page))
    )

    @sha256.validator
    def _validate_sha256(self, _attribute: str, value: str) -> None:
        if len(value) != 64 or not all(c in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid sha256 checksum: {value}")

    def to_dict(self) -> dict[str, str | list[dict[str, str | list[str] | int]] | None]:
        """Create a dict representation of the document, suitable for JSON serialization."""

        return dict(
            name=self.name,
            path=self.path,
            sha256=self.sha256,
            pages=[
                dict(
                    page_number=page.page_number,
                    text=page.text,
                    image_text=page.image_text,
                )
                for page in self.pages
            ],
        )

    @classmethod
    def from_textfile_bytes(cls, data: bytes, name: str, path: str | Path, encoding: str = "utf-8") -> ParsedDocument:
        return ParsedDocument(
            name=name,
            path=str(path),
            pages=[Page(page_number=0, text=data.decode(encoding).strip(), image_text=[])],
            sha256=hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def from_textfile(cls, path: Path, encoding: str = "utf-8") -> ParsedDocument:
        return cls.from_textfile_bytes(path.read_bytes(), name=path.name, path=path, encoding=encoding)

    @classmethod
    def from_dict(cls, data: dict[str, str | list[Any] | None]) -> ParsedDocument:
        """Create a ParsedDocument from a dict representation, as returned by to_dict()."""

        assert isinstance(data["name"], str) or data["name"] is None
        assert isinstance(data["path"], str)
        assert isinstance(data["sha256"], str)
        assert isinstance(data["pages"], list)

        return cls(
            name=data["name"],
            path=data["path"],
            sha256=data["sha256"],
            pages=[
                Page(
                    page_number=page["page_number"],
                    text=page["text"],
                    image_text=page["image_text"],
                )
                for page in data["pages"]
            ],
        )

    def get_snippets(self, images: bool = True) -> list[Snippet]:
        r"""
        Fetch the full texts of the document. That is:

        - the body of the document, pages joined with "\n\n"
        - each image's text, separately
        """

        snippets = []

        for snip in snippet_windows.windows_with_source(
            pages=[page for page in self.pages if page.text],
            join_str="\n\n",
            window_size=config.document_sync.snippet_window_size,
            step_size=config.document_sync.snippet_step_size,
        ):
            if len(snip.text) < config.document_sync.min_snippet_size:
                continue
            snippets.append(
                Snippet(
                    text=snip.text,
                    source_filename=f"{self.path}",
                    start_offset=snip.start,
                    page_start=snip.page_start,
                    page_end=snip.page_end,
                )
            )

        if not images:
            return snippets

        for page in self.pages:
            for image_text in page.image_text:
                if image_text:
                    for snip in snippet_windows.windows(
                        image_text,
                        window_size=config.document_sync.snippet_window_size,
                        step_size=config.document_sync.snippet_step_size,
                    ):
                        if len(snip.text) < config.document_sync.min_snippet_size:
                            continue
                        snippets.append(
                            Snippet(
                                text=snip.text,
                                source_filename=f"{self.path}",
                                start_offset=snip.start,
                                page_start=snip.page_start,
                                page_end=snip.page_end,
                            )
                        )
        return snippets
