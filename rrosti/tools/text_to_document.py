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

"""Consume a text file from stdin and output a JSON representation of a ParsedDocument."""

import argparse
import sys
from pathlib import Path

import orjson
from loguru import logger

from rrosti.snippets import parsed_document
from rrosti.snippets._page import Page
from rrosti.utils.misc import file_sha256


class ProgramArgs:
    input: list[Path]


def parse_args() -> ProgramArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Input file(s)", type=Path)
    return parser.parse_args(namespace=ProgramArgs())


def main() -> None:
    args = parse_args()
    for fname in args.input:
        logger.info(f"Parsing {fname}")
        with open(fname) as f:
            text = f.read().strip()
        doc = parsed_document.ParsedDocument(
            name=fname.name,
            path=str(fname),
            pages=[Page(page_number=0, text=text, image_text=[])],
            sha256=file_sha256(fname),
        )
        sys.stdout.buffer.write(orjson.dumps(doc.to_dict()))


if __name__ == "__main__":
    main()
