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

"""AST for a dialog state machine."""

from __future__ import annotations

import string
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

from overrides import override

from rrosti.utils.misc import JSONValue

MSG_CUT_LEN = 40


class JSONMixin:
    """Mixin for classes that can be converted to JSON."""

    @abstractmethod
    def to_json(self) -> JSONValue:
        """Convert to a dictionary that can be dumped as JSON or YAML."""


class GraphvizMixin(ABC):
    """Mixin for classes that can be converted to Graphviz dot format."""

    @abstractmethod
    def viz_label(self) -> str | None:
        """Return a label for the node in Graphviz dot string format."""


class ActionVisitor(ABC):  # noqa: B024 (intentional abstract class without abstract methods)
    """Visitor for actions."""

    def end_action(self, action: EndAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit an end action."""

    def goto_action(self, action: GotoAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a goto action."""

    def message_action(self, action: MessageAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a message action."""

    def send_action(self, action: SendAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a send action."""


class AsyncActionVisitor(ABC):  # noqa: B024 (intentional abstract class without abstract methods)
    """Visitor for actions."""

    async def end_action(self, action: EndAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit an end action."""

    async def goto_action(self, action: GotoAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a goto action."""

    async def message_action(self, action: MessageAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a message action."""

    async def send_action(self, action: SendAction) -> None:  # noqa: B027 (intentionally empty and not abstract)
        """Visit a send action."""


@dataclass
class Config:
    model_ast: str | None
    model: str | None = field(init=False)

    # TODO: This should be a non-dataclass and we should have a setter

    def __post_init__(self) -> None:
        self.model = self.model_ast

    # update the config with the given config, returning a new config
    def update(self, config: Config) -> Config:
        return Config(model_ast=config.model_ast or self.model_ast)

    def set_outer_config(self, outer: Config) -> None:
        self.model = self.model_ast or outer.model


class ASTNode(ABC):
    """Base class for all AST nodes."""

    _inherited_config: Config = Config(model_ast=None)

    @abstractproperty
    def _children(self) -> Sequence[ASTNode]:
        """Return a list of children."""

    def _link_configs(self, outer: Config) -> None:
        """Link the config to the outer config."""

        if hasattr(self, "config"):
            object.__setattr__(self, "_inherited_config", self.config)

        self._inherited_config.set_outer_config(outer)
        for child in self._children:
            child._link_configs(self._inherited_config)


class ASTLeaf(ASTNode):
    @property
    def _children(self) -> Sequence[ASTNode]:
        return []


class ActionBase(JSONMixin, GraphvizMixin, ASTNode, ABC):
    """Base class for actions."""

    @abstractproperty
    @override
    def _children(self) -> Sequence[ASTNode]:
        """Return the children of this node."""

    @abstractmethod
    def accept(self, visitor: ActionVisitor) -> None:
        """Accept a visitor."""

    @abstractmethod
    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        """Accept a visitor."""


class _VarExpansionFormatter(string.Formatter):
    """A string formatter that expands variables and records encountered function calls."""

    variables: dict[str, str]
    used_vars: set[str]
    placeholders: dict[str, str]

    def __init__(self, variables: dict[str, str]) -> None:
        super().__init__()
        self.variables = variables
        self.used_vars = set()
        self.placeholders = {}

    @override
    def get_value(self, key: int | str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        assert isinstance(key, str), "Only string keys are supported"
        assert len(args) == 0, "Positional arguments are not supported"
        assert len(kwargs) == 0, "Keyword arguments are not supported"
        if key in self.variables:
            self.used_vars.add(key)
            return self.variables[key]
        if key.endswith(")"):
            assert key.endswith("()"), key
            # add functions to undefineds so this can be redone with their values
            uuid = f"FUNCALL({uuid4()})"
            self.placeholders[uuid] = key
            return uuid
        raise ValueError(f"Undefined variable: {key}")


@dataclass
class _InterpolatedString:
    """
    Represents a string with variables interpolated.

    However, functions may not be resolved yet. Hence, they will be replaced with unique values.

    The `placeholders` field contains the mapping from the unique values to the function calls.
    """

    text: str
    placeholders: dict[str, str]


def _interpolate_vars(vars: dict[str, str], s: str, used_vars: frozenset[str] = frozenset([])) -> _InterpolatedString:
    """
    Interpolate variables in the variable definitions.

    Returns: Intepolated string on success, set of undefined variables on failure.
    """

    formatter = _VarExpansionFormatter(vars)
    formatted = formatter.format(s)
    if formatter.used_vars & used_vars:
        raise ValueError(f"Variable expansion loop detected: {formatter.used_vars & used_vars}")
    if not formatter.used_vars:
        return _InterpolatedString(formatted, formatter.placeholders.copy())

    news = _interpolate_vars(vars, formatted, used_vars | formatter.used_vars)
    news.placeholders.update(formatter.placeholders)
    return news


@dataclass(frozen=True)
class EndAction(ASTLeaf, ActionBase):
    """Represents the end of a conversation."""

    @override
    def to_json(self) -> JSONValue:
        return "end"

    def viz_label(self) -> str:
        return "end"

    @override
    def accept(self, visitor: ActionVisitor) -> None:
        visitor.end_action(self)

    @override
    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        await visitor.end_action(self)


@dataclass(frozen=True)
class GotoAction(ASTLeaf, ActionBase):
    """Represents an action that goes to another state in the state machine."""

    label: str

    @override
    def to_json(self) -> JSONValue:
        return {"goto": self.label}

    def viz_label(self) -> str:
        return f"GOTO {self.label}"

    @override
    def accept(self, visitor: ActionVisitor) -> None:
        visitor.goto_action(self)

    @override
    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        await visitor.goto_action(self)


def _interpolations(s: str) -> list[str]:
    """Get all interpolations in a message text."""

    return [fname for _, fname, _, _ in string.Formatter().parse(s) if fname is not None]


@dataclass(frozen=True)
class MessageAction(ASTLeaf, ActionBase):
    """Represents a message action, adding either a user or assistant message to the conversation."""

    # TODO: make the interpolation more robust and runnable only once (and automatically)

    role: Literal["user", "assistant", "system"]
    text: str
    placeholders: dict[str, str] = field(init=False)

    @override
    def to_json(self) -> JSONValue:
        if self.role == "user":
            return {"message": self.text}
        if self.role == "assistant":
            return {"assistant_message": self.text}
        if self.role == "system":
            return {"system_message": self.text}
        assert False, f"Invalid role: {self.role}"

    def viz_label(self) -> str:
        msg = self.text
        msg = msg.replace("\n", r"\n")
        if len(msg) > MSG_CUT_LEN:
            msg = msg[:MSG_CUT_LEN] + "..."
        return f'MSG({self.role}, "{msg}")'

    @override
    def accept(self, visitor: ActionVisitor) -> None:
        visitor.message_action(self)

    @override
    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        await visitor.message_action(self)

    def __post_init__(self) -> None:
        object.__setattr__(self, "placeholders", {})

    @property
    def interpolations(self) -> list[str]:
        """All interpolations in the message text."""
        return _interpolations(self.text)

    def _interpolate_vars(self, vars: dict[str, str]) -> None:
        """Interpolate variables in the message text."""
        s = _interpolate_vars(vars, self.text)
        object.__setattr__(self, "text", s.text)
        object.__setattr__(self, "placeholders", s.placeholders)


@dataclass(frozen=True)
class SendAction(ASTLeaf, ActionBase):
    """Represents a send action, sending a message (and transfering control) to another agent."""

    to: str
    next_state: str

    @override
    def to_json(self) -> JSONValue:
        return dict(send=dict(to=self.to, next_state=self.next_state))

    def viz_label(self) -> str:
        return f"SEND {self.to}, {self.next_state}"

    @override
    def accept(self, visitor: ActionVisitor) -> None:
        visitor.send_action(self)

    @override
    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        await visitor.send_action(self)


@dataclass(frozen=True)
class ActionList(JSONMixin, ASTNode):
    """Represents a list of actions to be executed in order."""

    actions: list[ActionBase]

    @property
    def _children(self) -> Sequence[ASTNode]:
        return self.actions

    def __post_init__(self) -> None:
        if len(self.actions) == 0:
            raise ValueError("Action list must contain at least one action")
        for action in self.actions[:-1]:
            if isinstance(action, GotoAction):
                raise ValueError("Only the last action may be a goto.")  # noqa: TRY004 (TypeError is not a good match)
            if isinstance(action, EndAction):
                raise ValueError("Only the last action may be an end.")  # noqa: TRY004 (TypeError is not a good match)
            if isinstance(action, SendAction):
                raise ValueError("Only the last action may be a send.")  # noqa: TRY004 (TypeError is not a good match)

    def target_state_name(self, curr: str | None = None) -> str | None:
        """Returns the target state of the action list, given the current state."""
        if isinstance(self.actions[-1], GotoAction):
            return self.actions[-1].label
        if isinstance(self.actions[-1], EndAction):
            return "END"
        return curr

    @override
    def to_json(self) -> JSONValue:
        return [a.to_json() for a in self.actions]

    def viz_label(self) -> str:
        actions = self.actions
        # If the last action is a goto or end, omit it
        if isinstance(actions[-1], (GotoAction, EndAction)):
            actions = actions[:-1]
        edge_labels = [a.viz_label() for a in actions]
        return "\n".join(label for label in edge_labels if label is not None)

    def accept(self, visitor: ActionVisitor) -> None:
        for action in self.actions:
            action.accept(visitor)

    async def aaccept(self, visitor: AsyncActionVisitor) -> None:
        for action in self.actions:
            await action.aaccept(visitor)

    def _interpolate_vars(self, vars: dict[str, str]) -> None:
        """Interpolate variables in the message text."""
        for action in self.actions:
            if isinstance(action, MessageAction):
                action._interpolate_vars(vars)


class ConditionBase(JSONMixin, ASTNode, ABC):
    """Base class for conditions."""

    action: ActionList

    config: Config

    def target_state(self, curr: str | None = None) -> str | None:
        """Get the target state of the condition. 'curr' is returned if there is no goto."""
        return self.action.target_state_name(curr)

    @abstractmethod
    def condition_label(self) -> str:
        """Get a human-readable label for the condition."""

    @abstractmethod
    def is_triggered_by(self, llm_output: str) -> bool:
        """Returns True if the condition is triggered by the given LLM output."""


@dataclass(frozen=True)
class Condition(ConditionBase):
    """Represents a condition that must be met for the action to be executed."""

    action: ActionList
    contains: str
    config: Config  # the config in the yaml, not necessarily the full config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return [self.action]

    @override
    def to_json(self) -> JSONValue:
        return {"contains": self.contains, "action": self.action.to_json()}

    @override
    def condition_label(self) -> str:
        return f'contains: "{self.contains}"'

    @override
    def is_triggered_by(self, llm_output: str) -> bool:
        return self.contains in llm_output


@dataclass(frozen=True)
class DefaultCondition(ConditionBase):
    """Represents a condition that met if no other conditions are met."""

    action: ActionList
    config: Config  # the config in the yaml, not necessarily the full config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return [self.action]

    @override
    def to_json(self) -> JSONValue:
        return {"default": self.action.to_json()}

    def viz_edge_label(self) -> str:
        return f"DEFAULT / {self.action.viz_label()}"

    @override
    def condition_label(self) -> str:
        return "default"

    @override
    def is_triggered_by(self, llm_output: str) -> bool:
        return True


class State(JSONMixin, ASTNode, ABC):
    """Base class for state machine states."""

    config: Config  # the config in the yaml, not necessarily the full config

    @abstractmethod
    def all_actions(self) -> list[ActionList]:
        """Returns all actions in the state."""

    def all_goto_state_labels(self) -> set[str]:
        """Returns all goto labels in the state."""
        s: set[str] = set()
        for action in self.all_actions():
            if isinstance(action.actions[-1], GotoAction):
                s.add(action.actions[-1].label)
            elif isinstance(action.actions[-1], SendAction):
                s.add(action.actions[-1].next_state)
        return s

    def all_send_agent_labels(self) -> set[str]:
        """Returns all send agent labels in the state."""
        s: set[str] = set()
        for action in self.all_actions():
            if isinstance(action.actions[-1], SendAction):
                s.add(action.actions[-1].to)
        return s


@dataclass(frozen=True)
class InitialState(State):
    """Represents the initial state of the state machine, without conditions."""

    action: ActionList
    config: Config  # the config in the yaml, not necessarily the full config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return [self.action]

    def __post_init__(self) -> None:
        for action in self.action.actions:
            if isinstance(action, SendAction):
                raise ValueError(  # noqa: TRY004 (TypeError is not a good match)
                    "Send action not allowed in initial state."
                )
        if not isinstance(self.action.actions[-1], GotoAction):
            raise ValueError("Initial state must end in a goto action.")  # noqa: TRY004 (TypeError is not a good match)

    @override
    def all_actions(self) -> list[ActionList]:
        return [self.action]

    @override
    def to_json(self) -> JSONValue:
        return {"name": "initial", "action": self.action.to_json()}


@dataclass(frozen=True)
class NonInitialState(State):
    """Represents any other but the initial state in the state machine"""

    name: str
    conditions: list[Condition]
    config: Config  # the config in the yaml, not necessarily the full config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return self.conditions

    def __post_init__(self) -> None:
        if self.name == "initial":
            raise ValueError('Attempt to create NonInitialState with name "initial".')
        if len(self.conditions) == 0:
            raise ValueError("State must contain at least one condition")
        for condition in self.conditions[:-1]:
            if isinstance(condition, DefaultCondition):
                raise ValueError(  # noqa: TRY004 (TypeError is not a good match)
                    "Only the last condition may be a default."
                )
        if not isinstance(self.conditions[-1], DefaultCondition):
            raise ValueError("The last condition must be a default.")  # noqa: TRY004 (TypeError is not a good match)
        if self.name in self.all_send_agent_labels():
            raise ValueError(f"State {self.name} sends to itself.")

    @override
    def all_actions(self) -> list[ActionList]:
        return [c.action for c in self.conditions]

    @override
    def to_json(self) -> JSONValue:
        return {
            "name": self.name,
            "conditions": [c.to_json() for c in self.conditions],
        }

    def triggered_condition(self, llm_output: str) -> Condition:
        for condition in self.conditions:
            if condition.is_triggered_by(llm_output):
                return condition
        # This should never happen since we are supposed to have a default condition
        raise ValueError(f"No condition triggered by {llm_output}")


@dataclass(frozen=True)
class Agent(JSONMixin, ASTNode):
    """Represents an agent."""

    name: str
    initial_state: InitialState
    noninitial_states: list[NonInitialState]
    _states_by_name: dict[str, NonInitialState] = field(init=False, repr=False)
    config: Config  # the config in the yaml, not necessarily the full config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return [self.initial_state, *self.noninitial_states]

    def get_state(self, name: str) -> NonInitialState:
        """Returns the state with the given name."""
        return self._states_by_name[name]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_states_by_name", {s.name: s for s in self.noninitial_states})

        # state names must be unique
        if len(self.noninitial_states) != len(self._states_by_name):
            raise ValueError("State names must be unique.")

        # no agent may send to itself
        if self.name in self.all_send_agent_labels():
            raise ValueError(f"Agent {self.name} sends to itself.")

    @override
    def to_json(self) -> JSONValue:
        return {
            "name": self.name,
            "states": [self.initial_state.to_json()] + [s.to_json() for s in self.noninitial_states],
        }

    def all_send_agent_labels(self) -> set[str]:
        """Returns all send agent labels in the agent."""
        s: set[str] = set()
        for state in self.noninitial_states:
            s.update(state.all_send_agent_labels())
        return s


@dataclass(frozen=True)
class Variables(JSONMixin):
    """Represents the variable definitions."""

    # variables are arbitrary identifiers mapped to strings. In yaml:
    # variables:
    #   foo: bar
    #   baz: quux
    vars: dict[str, str]

    @override
    def to_json(self) -> JSONValue:
        # (mypy does not support variance in mappings)
        return self.vars  # type: ignore[return-value]

    def __post_init__(self) -> None:
        for k, v in self.vars.items():
            s = _interpolate_vars(self.vars, v, frozenset([k]))
            if s.placeholders:
                raise ValueError(f"Undefined variables: {list(s.placeholders.values())}")
            self.vars[k] = s.text


@dataclass(frozen=True)
class StateMachine(ASTNode, JSONMixin):
    """Represents the conversation state machine."""

    variables: Variables
    agents: list[Agent]
    config: Config

    @property
    def _children(self) -> Sequence[ASTNode]:
        return self.agents

    def __post_init__(self) -> None:
        for agent in self.agents:
            for state in [agent.initial_state, *agent.noninitial_states]:
                for action_list in state.all_actions():
                    action_list._interpolate_vars(self.variables.vars)

        agent_names = [a.name for a in self.agents]
        if len(agent_names) != len(set(agent_names)):
            for name in agent_names:
                if agent_names.count(name) > 1:
                    raise ValueError(f"Duplicate agent name: {name}")
            assert False, "Unreachable"

        # unknown agents referenced?
        agent_references: set[str] = set()
        for agent in self.agents:
            agent_references.update(agent.all_send_agent_labels())
        for agent_reference in agent_references:
            if agent_reference not in agent_names:
                raise ValueError(f"Unknown agent: {agent_reference}")

        for child in self._children:
            child._link_configs(self.config)

    def get_agent(self, name: str) -> Agent:
        """Returns the agent with the given name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ValueError(f"No agent named {name}")

    @override
    def to_json(self) -> JSONValue:
        return {
            "variables": self.variables.to_json(),
            "agents": [a.to_json() for a in self.agents],
        }
