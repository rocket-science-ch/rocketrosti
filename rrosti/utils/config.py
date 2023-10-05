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

"""Configuration facility."""

import contextlib
import functools
import inspect
from pathlib import Path
from typing import IO, Any, Callable, Iterator, ParamSpec, TypeVar, cast

import attrs
import omegaconf
from attrs import validators as AV
from loguru import logger
from omegaconf import OmegaConf
from typing_extensions import Self

from rrosti.utils import misc

CONFIG_PATH = Path("config.yaml")
CONFIG_DEFAULTS_PATH = Path("assets/config.defaults.yaml")


@attrs.define
class FrontendConfig:
    websocket_url: str
    title: str


@attrs.define
class ModelCost:
    """Model usage costs per 1k tokens."""

    prompt_tokens: float
    completion_tokens: float = omegaconf.SI("${prompt_tokens}")

    def calculate(self, n_prompt_tokens: int, n_completion_tokens: int) -> float:
        return (n_prompt_tokens / 1000) * self.prompt_tokens + (n_completion_tokens / 1000) * self.completion_tokens


@attrs.define
class Endpoint:
    """Configuration items for an OpenAI API endpoint."""

    api_key: str = attrs.field(repr=False)
    max_embedding_requests_per_query: int
    engine_map: dict[str, str] | None = None
    api_base: str | None = None
    api_type: str | None = None
    api_version: str | None = None


@attrs.define
class OpenAI:
    embedding_model: str
    chat_completion_model: str
    max_tokens_per_model: dict[str, int]
    completion_max_tokens: int = attrs.field(validator=AV.gt(0))

    # We refuse to query if we can request fewer tokens than this
    completion_min_tokens: int = attrs.field(validator=AV.gt(0))

    completion_temperature: float = attrs.field(validator=AV.ge(0.0))
    chat_models: list[str]
    model_cost: dict[str, ModelCost]
    use_endpoint: str
    endpoints: dict[str, Endpoint]

    num_chat_completion_worker_threads: int = attrs.field(validator=AV.gt(0))
    num_embedding_worker_threads: int = attrs.field(validator=AV.gt(0))

    rate_limit_initial_interval: float = attrs.field(validator=AV.gt(0.0))
    rate_limit_min_interval: float = attrs.field(validator=AV.gt(0.0))
    rate_limit_max_interval: float = attrs.field(validator=AV.gt(0.0))
    rate_limit_error_multiplier: float = attrs.field(validator=AV.gt(1.0))
    rate_limit_success_multiplier: float = attrs.field(validator=AV.gt(0.0))
    rate_limit_decrease_after_seconds: float = attrs.field(validator=AV.gt(0.0))

    @property
    def endpoint(self) -> Endpoint:
        return self.endpoints[self.use_endpoint]

    def __attrs_post_init__(self) -> None:
        assert self.completion_min_tokens <= self.completion_max_tokens
        assert self.chat_completion_model in self.chat_models


@attrs.define
class DocumentSync:
    data_gen_path: Path
    source_docs_path: Path
    parsed_docs_path: Path
    snippet_window_size: int
    snippet_step_size: int
    min_snippet_size: int


@attrs.define
class StateMachine:
    yaml_path: Path = attrs.field()
    debug_detect_unresolved_funcalls: bool
    rtfm_max_tokens: int = attrs.field(validator=AV.gt(0))
    rtfm_merge_candidates: int = attrs.field(validator=AV.gt(0))

    @yaml_path.validator
    def _check_yaml_path(self, _attribute: str, value: Path) -> None:
        if not value.exists():
            raise ValueError(f"YAML file {value} does not exist")


@attrs.define
class Backend:
    listen_port: int


@attrs.define
class _Config:
    backend: Backend
    document_sync: DocumentSync
    frontend: FrontendConfig
    openai_api: OpenAI
    state_machine: StateMachine


@contextlib.contextmanager
def _with_resolver(resolver: str, func: Callable[..., str]) -> Iterator[None]:
    OmegaConf.register_new_resolver(resolver, func)
    try:
        yield
    finally:
        OmegaConf.clear_resolver(resolver)


class ConfigurationNotLoadedError(Exception):
    pass


class _ConfigProxy:
    global config

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return super().__getattribute__(name)

        # This can happen if this has been "imported from" before the config was loaded
        if isinstance(config, _Config):
            return getattr(config, name)

        logger.error("Configuration not loaded yet")
        raise ConfigurationNotLoadedError("Configuration not loaded yet")


config: _Config = cast(_Config, _ConfigProxy())


class _ConfigContext:
    old_value: _Config
    new_value: _Config

    def __init__(self, new_value: _Config) -> None:
        global config

        logger.info("Config context: old={}({}), new={}({})", type(config), id(config), type(new_value), id(new_value))
        self.old_value = config
        self.new_value = new_value
        config = new_value

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        global config
        if config is not self.new_value:
            logger.warning("Config context: config changed during context; not restoring")
            return
        logger.info("Config context: restoring config to {}({})", type(self.old_value), id(self.old_value))
        config = self.old_value


def load(
    *,
    reload: bool = False,
    defaults_file: Path | IO[Any] = CONFIG_DEFAULTS_PATH,
    config_file: Path | IO[Any] | None = CONFIG_PATH,
    allow_module_level: bool = False,
) -> _ConfigContext:
    """
    Load the configuration.

    Do not call from module level (at import time).
    """

    global config

    if isinstance(config, _Config) and not reload:
        logger.warning("Configuration already loaded")
        return _ConfigContext(config)

    if not allow_module_level and misc.is_import_in_progress():
        raise ConfigurationNotLoadedError("Configuration loaded during import")

    def _read_file(fname: str, default: str) -> str:
        path = Path(fname).expanduser()
        if path.exists():
            return path.read_text().strip()
        return default

    with _with_resolver("file", _read_file):
        cfg = OmegaConf.structured(_Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.load(defaults_file))

        do_load_config = config_file is not None
        if isinstance(config_file, Path) and not config_file.exists():
            logger.warning("Configuration file {} does not exist", config_file)
            do_load_config = False

        if do_load_config:
            logger.info("Loading configuration from {}", config_file)
            assert config_file is not None
            cfg = OmegaConf.merge(cfg, OmegaConf.load(config_file))
        else:
            logger.info("Not loading config: {}", config_file)

        cfg = OmegaConf.to_object(cfg)

    assert isinstance(cfg, _Config)

    # Some further validation
    assert (
        cfg.openai_api.completion_max_tokens
        <= cfg.openai_api.max_tokens_per_model[cfg.openai_api.chat_completion_model]
    )

    return _ConfigContext(cfg)


def load_test_config(config_file: Path | IO[Any] = Path("/nonexistent")) -> _ConfigContext:
    ctx = load(config_file=config_file, allow_module_level=True, reload=True)
    assert isinstance(config, _Config)
    return ctx


@attrs.frozen
class _FromConfig:
    key: str


FromConfig: Callable[[str], Any] = _FromConfig


@functools.cache
def _get_config_item(path: str) -> Any:
    cfg = config
    for part in path.split("."):
        cfg = getattr(cfg, part)
    return cfg


_T = TypeVar("_T")


def _check_config_key(path: str) -> None:
    cls: type[Any] = _Config
    for part in path.split("."):
        with contextlib.suppress(attrs.exceptions.NotAnAttrsClassError):
            cls = getattr(attrs.fields(cls), part).type


_Params = ParamSpec("_Params")


def uses_config(func: Callable[_Params, _T]) -> Callable[_Params, _T]:
    """Replace uses of FromConfig by values from config."""

    config_args: dict[str, Any] | None = None

    # Validate the config keys
    for arg_value in inspect.signature(func).parameters.values():
        if isinstance(arg_value.default, _FromConfig):
            _check_config_key(arg_value.default.key)

    def init() -> None:
        nonlocal config_args

        if misc.is_import_in_progress():
            raise ConfigurationNotLoadedError("Function with @uses_config decorator called during import")

        config_args = {}
        for arg_name, arg_value in inspect.signature(func).parameters.items():
            if isinstance(arg_value.default, _FromConfig):
                config_args[arg_name] = _get_config_item(arg_value.default.key)

    @functools.wraps(func)
    def wrapper(*args: _Params.args, **kwargs: _Params.kwargs) -> _T:
        if config_args is None:
            init()
        assert config_args is not None
        kwargs = {**config_args, **kwargs}  # type: ignore[assignment]
        return func(*args, **kwargs)

    return wrapper
