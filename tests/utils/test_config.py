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

import io
from typing import Iterable

import pytest
from pytest_mock import MockerFixture

from rrosti.utils import config

TEST_CONFIG = r"""
backend:
    listen_port: 314
"""


@pytest.fixture
def _mock_config(mocker: MockerFixture) -> Iterable[None]:
    with config.load_test_config(config_file=io.StringIO(TEST_CONFIG)):
        yield


@pytest.mark.usefixtures("_mock_config")
def test_default_values_from_config() -> None:
    @config.uses_config
    def example_function(arg1: int = config.FromConfig("backend.listen_port")) -> None:
        assert arg1 == 314

    example_function()


@pytest.mark.usefixtures("_mock_config")
def test_invalid_config_variable() -> None:
    def example_function(arg1: int = config.FromConfig("backend.loguru_rotation_size_megabytes2")) -> None:
        pass

    with pytest.raises(AttributeError):
        config.uses_config(example_function)


@pytest.mark.usefixtures("_mock_config")
def test_config_not_loaded_at_wrapper(mocker: MockerFixture) -> None:
    config.CONFIG_PATH = config.CONFIG_PATH.parent / "config.invalid.yaml"

    @config.uses_config
    def example_function(arg1: int = config.FromConfig("backend.listen_port")) -> None:
        assert arg1 == 314

    example_function()
    config.CONFIG_PATH = config.CONFIG_PATH.parent / "config.defaults.yaml"
