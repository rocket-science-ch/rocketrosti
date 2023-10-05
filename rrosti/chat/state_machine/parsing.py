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

"""Read a description of a dialog state machine from a .yaml file and constructs an AST."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TextIO, cast

import jsonschema
import marshmallow as ma
from loguru import logger
from overrides import override
from ruamel.yaml import YAML

from rrosti.chat.state_machine import ast

yaml = YAML(typ=["rt", "string"])

# We don't really use this for anything but validating, but it helps us give better error messages.
_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "variables": {
            "type": "object",
            "additionalProperties": {"type": "string"},  # Variables are arbitrary string values
        },
        "agents": {
            "type": "array",
            "items": {"$ref": "#/definitions/agent"},
            "minItems": 1,
        },
        "config": {
            "model": {"type": "string"},
        },
        "required": ["agents"],
    },
    "required": ["agents"],
    "definitions": {
        "initial_state": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"const": "initial"},
                "action": {
                    "$ref": "#/definitions/action_list",
                },
                "model": {"type": "string"},
            },
            "required": ["name", "action"],
        },
        "noninitial_state": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "not": {"const": "initial"}},
                "conditions": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/condition"},
                },
                "model": {"type": "string"},
            },
            "required": ["name", "conditions"],
        },
        "states_list": {
            "type": "array",
            "minItems": 1,  # Must have at least one state
            "prefixItems": [
                {"$ref": "#/definitions/initial_state"},
            ],
            "items": {
                "$ref": "#/definitions/noninitial_state",
            },
        },
        "agent": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "states": {
                    "$ref": "#/definitions/states_list",
                },
                "model": {"type": "string"},
            },
            "required": ["name", "states"],
        },
        "atomic_actions": {
            "end": {"const": "end"},
            "goto": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "goto": {"type": "string"},
                },
                "required": ["goto"],
            },
            "message": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
            "assistant_message": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "assistant_message": {"type": "string"},
                },
                "required": ["assistant_message"],
            },
            "system_message": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "system_message": {"type": "string"},
                },
                "required": ["system_message"],
            },
            "send": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "send": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "to": {"type": "string"},
                            "next_state": {"type": "string"},
                        },
                    },
                },
                "required": ["send"],
            },
        },
        "action": {
            "oneOf": [
                {"$ref": "#/definitions/atomic_actions/end"},
                {"$ref": "#/definitions/atomic_actions/goto"},
                {"$ref": "#/definitions/atomic_actions/message"},
                {"$ref": "#/definitions/atomic_actions/assistant_message"},
                {"$ref": "#/definitions/atomic_actions/system_message"},
                {"$ref": "#/definitions/atomic_actions/send"},
            ],
        },
        "action_list": {
            "type": "array",
            "items": {"$ref": "#/definitions/action"},
            "minItems": 1,
        },
        "if_cond_expr": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "contains": {"type": "string"},
            },
            "required": ["contains"],
        },
        "then_object": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model": {"type": "string"},
                "action": {"$ref": "#/definitions/action_list"},
            },
            "required": ["action"],
        },
        "if_then": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "if": {"$ref": "#/definitions/if_cond_expr"},
                "then": {"$ref": "#/definitions/then_object"},
            },
            "required": ["if", "then"],
        },
        "default": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "default": {"$ref": "#/definitions/then_object"},
            },
            "required": ["default"],
        },
        "condition": {
            "oneOf": [
                {"$ref": "#/definitions/default"},
                {"$ref": "#/definitions/if_then"},
            ],
        },
    },
}


def _parse_action_one_param(dic: Mapping[str, Any]) -> ast.ActionBase:
    assert len(dic) == 1, f"Invalid action: {dic}"
    key, value = next(iter(dic.items()))
    assert isinstance(value, str), f"Invalid action: {dic}"
    assert isinstance(key, str), f"Invalid action: {dic}"
    value = value.strip()
    match key:
        case "goto":
            return ast.GotoAction(value)
        case "message":
            return ast.MessageAction("user", value)
        case "assistant_message":
            return ast.MessageAction("assistant", value)
        case "system_message":
            return ast.MessageAction("system", value)
        case _:
            raise ma.ValidationError(f"Invalid action: {dic}")


def _parse_action(value: Any) -> ast.ActionBase:
    match value:
        case "end":
            return ast.EndAction()
        case str():
            raise ma.ValidationError(f"Invalid action: {value}")
        case {"send": dict() as send}:
            return ast.SendAction(send["to"], send["next_state"])
        case dict():
            return _parse_action_one_param(value)
        case _:
            raise ma.ValidationError(f"Invalid action: {value}")


class ActionListField(ma.fields.Field):
    """
    Parse a list of actions of the format
    ["end", {"goto": "label"}, {"message": "text"}, {"assistant_message": "text"}, ...]
    """

    @override
    def _deserialize(
        self, value: Any, attr: str | None, data: Mapping[str, Any] | None, **kwargs: Any
    ) -> ast.ActionList:
        logger.info("ActionListField: value={}, attr={}, data={}, kwargs={}", value, attr, data, kwargs)
        if not isinstance(value, list):
            raise ma.ValidationError(f"Invalid action: {value}")
        try:
            return ast.ActionList([_parse_action(v) for v in value])
        except ValueError as e:
            raise ma.ValidationError(*e.args) from e


class ConfigSchema(ma.Schema):
    model = ma.fields.String(required=False)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_config(self, data: dict[str, Any], **kwargs: Any) -> ast.Config:
        logger.info("make_config: data={}, kwargs={}", data, kwargs)
        return ast.Config(model_ast=data.get("model", None))


class IfSchema(ma.Schema):
    contains = ma.fields.String(required=True)

    class Meta:
        unknown = ma.EXCLUDE


class ThenSchema(ma.Schema):
    action = ActionListField(required=True)
    model = ma.fields.String(required=False)

    class Meta:
        unknown = ma.EXCLUDE


class IfThenSchema(ma.Schema):
    if_ = ma.fields.Nested(IfSchema, required=True, data_key="if")
    then = ma.fields.Nested(ThenSchema, required=True)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_if_then(self, data: dict[str, Any], **kwargs: Any) -> ast.Condition:
        logger.info("make_if_then: data={}, kwargs={}", data, kwargs)
        return ast.Condition(
            contains=data["if"]["contains"],
            action=data["then"]["action"],
            config=ast.Config(model_ast=data["then"].get("model")),
        )


class DefaultSchema(ma.Schema):
    # default = ma.fields.Nested(ThenSchema, required=True)
    action = ActionListField(required=True)
    model = ma.fields.String(required=False)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_default(self, data: dict[str, Any], **kwargs: Any) -> ast.DefaultCondition:
        logger.info("make_default: data={}, kwargs={}", data, kwargs)
        return ast.DefaultCondition(action=data["action"], config=ast.Config(model_ast=data.get("model")))


class ConditionSchema(ma.Schema):
    # We have either:
    # if:
    #     contains: "foo"
    # then:
    #     action: ["end", {"goto": "label"}, ...]
    #     model: gpt-4   # optional
    #
    # OR:
    #
    # default:
    #     action: ["end", {"goto": "label"}, ...]
    #     model: gpt-4   # optional
    if_ = ma.fields.Nested(IfSchema, required=False, data_key="if")
    then = ma.fields.Nested(ThenSchema, required=False)
    default = ma.fields.Nested(DefaultSchema, required=False)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_condition(self, data: dict[str, Any], **kwargs: Any) -> ast.Condition | ast.DefaultCondition:
        logger.info("make_condition: data={}, kwargs={}", data, kwargs)
        if "if_" in data:
            assert "default" not in data, "Cannot have both if and default"
            assert "then" in data, "if without then"
            return ast.Condition(
                contains=data["if_"]["contains"],
                action=data["then"]["action"],
                config=ast.Config(model_ast=data["then"].get("model")),
            )
        assert data.get("default"), "Must have either if or default"
        assert "then" not in data, "default cannot have then"
        return data["default"]  # type: ignore[no-any-return]


class StateSchema(ma.Schema):
    name = ma.fields.String(required=True)
    model = ma.fields.String(required=False)

    # Conditions are mandatory for non-initial states and forbidden for initial states.
    conditions = ma.fields.List(ma.fields.Nested(ConditionSchema), required=False)
    # The opposite applies to action.
    action = ActionListField(required=False)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_state(self, data: dict[str, Any], **kwargs: Any) -> ast.InitialState | ast.NonInitialState:
        logger.info("make_state: data={}, kwargs={}", data, kwargs)
        if data["name"] == "initial":
            assert "conditions" not in data, "Initial state cannot have conditions"
            assert "action" in data, "Initial state must have action"
            try:
                return ast.InitialState(action=data["action"], config=ast.Config(model_ast=data.get("model")))
            except ValueError as e:
                raise ma.ValidationError(*e.args) from e
        assert "conditions" in data, "Non-initial state must have conditions"
        assert "action" not in data, "Non-initial state cannot have action"
        try:
            return ast.NonInitialState(
                name=data["name"], conditions=data["conditions"], config=ast.Config(model_ast=data.get("model"))
            )
        except ValueError as e:
            raise ma.ValidationError(*e.args) from e


def _validate_states(states: list[ast.State]) -> None:
    """Validate that all state names are unique."""
    noninitials: list[ast.NonInitialState]

    match states:
        case []:
            raise ma.ValidationError("There must be at least one state")
        case [ast.InitialState(), *noninitials_]:
            if not all(isinstance(s, ast.NonInitialState) for s in noninitials_):
                raise ma.ValidationError("Only the first state can be initial")
            noninitials = cast(list[ast.NonInitialState], noninitials_)
        case [ast.NonInitialState(), *_]:
            raise ma.ValidationError("The first state must be initial")
        case _:
            assert False, "Unreachable"

    names = [s.name for s in noninitials]
    if any(s == "initial" for s in names):
        raise ma.ValidationError("State name 'initial' is reserved")
    if len(set(names)) != len(names):
        dups = {name for name in names if names.count(name) > 1}
        raise ma.ValidationError(f"Duplicate state name: {dups if len(dups) != 1 else dups.pop()}")

    # all gotoed states must exist
    for state in states:
        for label in state.all_goto_state_labels():
            if label not in names:
                raise ma.ValidationError(f"State '{label}' does not exist")

    # all states must be reachable
    reachable = {"initial"}
    reachable.update(states[0].all_goto_state_labels())
    while True:
        new_reachable = reachable.copy()
        for state in noninitials:
            if state.name in reachable:
                new_reachable.update(state.all_goto_state_labels())
        if new_reachable == reachable:
            break
        reachable = new_reachable
    unreachable = set(names) - reachable
    if unreachable:
        raise ma.ValidationError(f"States {unreachable} are unreachable")


class AgentSchema(ma.Schema):
    name = ma.fields.String(required=True)
    states = ma.fields.List(ma.fields.Nested(StateSchema), required=True, validate=_validate_states)
    model = ma.fields.String(required=False)

    class Meta:
        unknown = ma.EXCLUDE

    @ma.post_load
    def make_agent(self, data: dict[str, Any], **kwargs: Any) -> ast.Agent:
        logger.info("make_agent: data={}, kwargs={}", data, kwargs)
        states = data["states"]
        assert len(states) >= 1, "There must be at least one state"
        try:
            return ast.Agent(
                name=data["name"],
                initial_state=states[0],
                noninitial_states=states[1:],
                config=ast.Config(model_ast=data.get("model")),
            )
        except ValueError as e:
            raise ma.ValidationError(*e.args) from e


class StateMachineSchema(ma.Schema):
    variables = ma.fields.Dict(keys=ma.fields.String(), values=ma.fields.String(), required=False)
    agents = ma.fields.List(ma.fields.Nested(AgentSchema), required=True)
    config = ma.fields.Nested(ConfigSchema, required=False)

    class Meta:
        unknown = ma.EXCLUDE

    # Use jsonschema to validate the dict
    @ma.pre_load
    def validate(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        v = jsonschema.Draft202012Validator(_SCHEMA)
        errors = sorted(v.iter_errors(data), key=lambda e: e.path)
        if errors:
            for e in errors:
                logger.error("Validation error:\n{}", e)
            raise ma.ValidationError("\n-------\n".join(str(e) for e in errors))
        return data

    @ma.post_load
    def make_state_machine(self, data: dict[str, Any], **kwargs: Any) -> ast.StateMachine:
        variables = data.get("variables", {})
        variables = {k: v.strip() for k, v in variables.items()}
        config = data.get("config", ast.Config(model_ast=None))

        try:
            return ast.StateMachine(variables=ast.Variables(variables), agents=data["agents"], config=config)
        except ValueError as e:
            raise ma.ValidationError(*e.args) from e


state_machine_schema = StateMachineSchema()


def loads_from_yaml(s: str) -> ast.StateMachine:
    """Load a state machine from a YAML string."""
    d = yaml.load(s)
    sm: ast.StateMachine = state_machine_schema.load(d)
    return sm


def load_from_yaml(f: TextIO) -> ast.StateMachine:
    """Load a state machine from a YAML file."""
    d = yaml.load(f)
    sm: ast.StateMachine = state_machine_schema.load(d)
    return sm


def load_from_yaml_file(path: Path) -> ast.StateMachine:
    """Load a state machine from a YAML file."""
    with open(path) as f:
        return load_from_yaml(f)
