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

from pprint import pformat

import pytest
from loguru import logger
from marshmallow import ValidationError

import rrosti.utils.config
from rrosti.chat.state_machine import ast, parsing
from rrosti.utils.config import config

# This loads the default config
rrosti.utils.config.load_test_config()


def test_minimal_example() -> None:
    dsm = parsing.loads_from_yaml(
        """
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
        """
    )

    assert isinstance(dsm, ast.StateMachine)
    assert isinstance(dsm.variables, ast.Variables)
    assert dsm.variables.vars == {}
    assert len(dsm.agents) == 1
    assert dsm.agents[0].name == "agent1"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert len(agent.noninitial_states) == 1

    state = agent.noninitial_states[0]
    assert isinstance(state, ast.NonInitialState)
    assert state.name == "some_state"
    assert len(state.conditions) == 1

    cond = state.conditions[0]
    assert isinstance(cond, ast.DefaultCondition)
    assert cond.action == ast.ActionList([ast.EndAction()])


def test_goto_does_not_exist() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: nonexistent_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert "State 'nonexistent_state' does not exist" in str(excinfo.value)


def test_goto() -> None:
    dsm = parsing.loads_from_yaml(
        """
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    assert isinstance(dsm.variables, ast.Variables)
    assert dsm.variables.vars == {}
    assert len(dsm.agents) == 1
    assert dsm.agents[0].name == "agent1"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert len(agent.noninitial_states) == 1

    state = agent.noninitial_states[0]
    assert isinstance(state, ast.NonInitialState)
    assert state.name == "some_state"
    assert len(state.conditions) == 1

    cond = state.conditions[0]
    assert isinstance(cond, ast.DefaultCondition)
    assert cond.action == ast.ActionList([ast.EndAction()])


def test_undefined_variable() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   message: "{undefined_variable}"
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert "Undefined variable" in str(excinfo.value)


def test_expand_variable_in_variable() -> None:
    dsm = parsing.loads_from_yaml(
        """
    variables:
        variable1: "default value"
        variable2: "{variable1}"
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   message: "{variable2}"
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    assert isinstance(dsm.variables, ast.Variables)
    assert dsm.variables.vars == {"variable1": "default value", "variable2": "default value"}
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList(
        [ast.MessageAction(role="user", text="default value"), ast.GotoAction(label="some_state")]
    )


def test_self_referencing_variable() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        variables:
            variable1: "{variable1}"
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   system_message: "Hello system!"
                -   message: "{variable1}"
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
            """
        )

    assert "Variable expansion loop detected" in str(excinfo.value)


def test_circular_variable() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        variables:
            variable1: "{variable2}"
            variable2: "{variable1}"
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   message: "{variable1}"
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert "Variable expansion loop detected" in str(excinfo.value)


def test_variable() -> None:
    dsm = parsing.loads_from_yaml(
        """
    variables:
        variable1: "default value"
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   message: "{variable1}"
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    assert isinstance(dsm.variables, ast.Variables)
    assert dsm.variables.vars == {"variable1": "default value"}
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList(
        [ast.MessageAction(role="user", text="default value"), ast.GotoAction(label="some_state")]
    )
    assert len(agent.noninitial_states) == 1

    state = agent.noninitial_states[0]
    assert isinstance(state, ast.NonInitialState)
    assert state.name == "some_state"
    assert len(state.conditions) == 1

    cond = state.conditions[0]
    assert isinstance(cond, ast.DefaultCondition)
    assert cond.action == ast.ActionList([ast.EndAction()])


def test_state_not_reachable() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   end
            -   name: unreachable_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert " are unreachable" in str(excinfo.value)


# Also test with two unreachable states that are connected to each other
def test_state_not_reachable2() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state1
            -   name: unreachable_state1
                conditions:
                -   default:
                        action:
                        -   goto: unreachable_state2
            -   name: unreachable_state2
                conditions:
                -   default:
                        action:
                        -   goto: unreachable_state1
        """
        )

    assert " are unreachable" in str(excinfo.value)


def test_duplicate_state_name() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state2
            -   name: state2
                conditions:
                -   default:
                        action:
                        -   goto: state3
            -   name: state3
                conditions:
                -   default:
                        action:
                        -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state2
        """
        )

    assert "Duplicate state name: state1" in str(excinfo.value)


def test_duplicate_agent_name() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state2
            -   name: state2
                conditions:
                -   default:
                        action:
                        -   goto: state3
            -   name: state3
                conditions:
                -   default:
                        action:
                        -   goto: state1
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: state1
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state2
            -   name: state2
                conditions:
                -   default:
                        action:
                        -   goto: state3
            -   name: state3
                conditions:
                -   default:
                        action:
                        -   goto: state1
        """
        )

    assert "Duplicate agent name: agent1" in str(excinfo.value)


def test_first_state_not_initial() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: state1
                conditions:
                -   default:
                        action:
                        -   goto: state2
            -   name: state2
                conditions:
                -   default:
                        action:
                        -   goto: state3
            -   name: state3
                conditions:
                -   default:
                        action:
                        -   goto: state1
        """
        )

    # assert "The first state must be initial" in str(excinfo.value)
    assert "'initial' was expected" in str(excinfo.value)


def test_no_states() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   states: []
        """
        )

    # assert "There must be at least one state" in str(excinfo.value)
    assert "[] is too short" in str(excinfo.value)


def test_send_action() -> None:
    dsm = parsing.loads_from_yaml(
        """
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   send:
                            to: agent2
                            next_state: some_state
    -   name: agent2
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   send:
                            to: agent1
                            next_state: some_state
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert agent._states_by_name["some_state"].conditions[0].action == ast.ActionList(
        [
            ast.SendAction(
                to="agent2",
                next_state="some_state",
            )
        ]
    )


def test_state_with_model() -> None:
    dsm = parsing.loads_from_yaml(
        """
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            model: some_model
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert agent._states_by_name["some_state"].config.model_ast == "some_model"


def test_if_then() -> None:
    dsm = parsing.loads_from_yaml(
        """
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   message: some_message
            -   goto: some_state
        -   name: some_state
            conditions:
            -   if:
                    contains: some_substring
                then:
                    model: some_model
                    action:
                    -   goto: some_state
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList(
        [
            ast.MessageAction(text="some_message", role="user"),
            ast.GotoAction("some_state"),
        ]
    )
    assert len(agent._states_by_name["some_state"].conditions) == 2
    assert agent._states_by_name["some_state"].conditions[0].action == ast.ActionList(
        [
            ast.GotoAction("some_state"),
        ]
    )
    assert agent._states_by_name["some_state"].conditions[0].config.model_ast == "some_model"
    assert agent._states_by_name["some_state"].conditions[0].config.model == "some_model"
    assert agent._states_by_name["some_state"].conditions[1].action == ast.ActionList(
        [
            ast.EndAction(),
        ]
    )
    assert agent._states_by_name["some_state"].conditions[1].config.model_ast is None
    assert agent._states_by_name["some_state"].conditions[1].config.model is None


def test_send_to_self_is_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   send:
                                to: agent1
                                next_state: some_state
        """
        )

    assert "Agent agent1 sends to itself." in str(excinfo.value)


def test_send_to_unknown_agent_is_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   send:
                                to: agent2
                                next_state: some_state
        """
        )

    assert "Unknown agent: agent2" in str(excinfo.value)


def test_send_to_unknown_state_is_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   send:
                                to: agent1
                                next_state: some_state2
        """
        )

    assert "State 'some_state2' does not exist" in str(excinfo.value)


def test_send_must_be_last_action() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   message: some_message
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   send:
                                to: agent2
                                next_state: some_state
                        -   end
        -   name: agent2
            states:
            -   name: initial
                action:
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert "Only the last action may be a send." in str(excinfo.value)


def test_send_not_allowed_in_initial() -> None:
    with pytest.raises(ValidationError) as excinfo:
        parsing.loads_from_yaml(
            """
        agents:
        -   name: agent1
            states:
            -   name: initial
                action:
                -   send:
                        to: agent2
                        next_state: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        -   name: agent2
            states:
            -   name: initial
                action:
                -   goto: some_state
            -   name: some_state
                conditions:
                -   default:
                        action:
                        -   end
        """
        )

    assert "Send action not allowed in initial state" in str(excinfo.value)


def test_model_in_global_config() -> None:
    dsm = parsing.loads_from_yaml(
        """
    config:
        model: some_model
    agents:
    -   name: agent1
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    assert isinstance(dsm, ast.StateMachine)
    assert dsm.config.model_ast == "some_model"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert agent._states_by_name["some_state"].config.model_ast is None
    assert agent._states_by_name["some_state"].config.model == "some_model"


def test_override_model_in_agent() -> None:
    dsm = parsing.loads_from_yaml(
        """
    config:
        model: some_model
    agents:
    -   name: agent1
        model: some_model2
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    action:
                    -   end
    """
    )

    logger.info("dsm:\n{}", pformat(dsm))

    assert isinstance(dsm, ast.StateMachine)
    assert dsm.config.model_ast == "some_model"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert agent.config.model_ast == "some_model2"
    assert agent.config.model == "some_model2"
    assert agent._states_by_name["some_state"].config.model == "some_model2"


def test_override_model_in_condition() -> None:
    dsm = parsing.loads_from_yaml(
        """
    config:
        model: some_model
    agents:
    -   name: agent1
        model: some_model2
        states:
        -   name: initial
            action:
            -   goto: some_state
        -   name: some_state
            conditions:
            -   default:
                    model: some_model3
                    action:
                    -   end
    """
    )

    logger.info("dsm:\n{}", pformat(dsm))

    assert isinstance(dsm, ast.StateMachine)
    assert dsm.config.model_ast == "some_model"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.action == ast.ActionList([ast.GotoAction(label="some_state")])
    assert agent.config.model_ast == "some_model2"
    assert agent.config.model == "some_model2"
    assert agent._states_by_name["some_state"].config


def test_model_in_initial() -> None:
    dsm = parsing.loads_from_yaml(
        """
config:
    model: some_model
agents:
-   name: agent1
    states:
    -   name: initial
        model: some_model2
        action:
        -   goto: some_state
    -   name: some_state
        conditions:
        -   default:
                action:
                -   end
"""
    )
    assert isinstance(dsm, ast.StateMachine)
    assert dsm.config.model_ast == "some_model"
    agent = dsm.agents[0]
    assert isinstance(agent.initial_state, ast.InitialState)
    assert agent.initial_state.config.model_ast == "some_model2"


def test_model_thorough() -> None:
    dsm = parsing.loads_from_yaml(
        """
config:
    model: model_global
agents:
-   name: agent_1
    states:
    -   name: initial
        model: model_initial
        action:
        -   message: x
        -   goto: main
    -   name: main
        conditions:
        -   if:
                contains: 'cond1'
            then:
                model: model_cond1
                action:
                -   message: 'user_input'
        -   if:
                contains: 'cond3'
            then:
                model: model_cond3
                action:
                -   send:
                        to: agent_2
                        next_state: main
        -   default:
                model: model_default
                action:
                -   message: "user_input"
-   name: agent_2
    states:
    -   name: initial
        model: model_initial2
        action:
        -   message: "x"
        -   goto: main
    -   name: main
        conditions:
        -   if:
                contains: 'cond6'
            then:
                model: model_cond6
                action:
                -   send:
                        to: agent_1
                        next_state: main
        -   default:
                model: model_default2
                action:
                -   message: "x"
"""
    )
    assert isinstance(dsm, ast.StateMachine)
    assert dsm.config.model_ast == "model_global"
    agent1 = dsm.agents[0]
    assert agent1.initial_state.config.model_ast == "model_initial"
    assert agent1.initial_state.action == ast.ActionList(
        [
            ast.MessageAction(text="x", role="user"),
            ast.GotoAction(label="main"),
        ]
    )
    assert agent1._states_by_name["main"].conditions[0].config.model_ast == "model_cond1"
    assert agent1._states_by_name["main"].conditions[0].action == ast.ActionList(
        [
            ast.MessageAction(text="user_input", role="user"),
        ]
    )
    assert agent1._states_by_name["main"].conditions[1].config.model_ast == "model_cond3"
    assert agent1._states_by_name["main"].conditions[1].action == ast.ActionList(
        [
            ast.SendAction(to="agent_2", next_state="main"),
        ]
    )
    assert agent1._states_by_name["main"].conditions[2].config.model_ast == "model_default"
    assert agent1._states_by_name["main"].conditions[2].action == ast.ActionList(
        [
            ast.MessageAction(text="user_input", role="user"),
        ]
    )

    agent2 = dsm.agents[1]
    assert agent2.initial_state.config.model_ast == "model_initial2"
    assert agent2.initial_state.action == ast.ActionList(
        [
            ast.MessageAction(text="x", role="user"),
            ast.GotoAction(label="main"),
        ]
    )
    assert agent2._states_by_name["main"].conditions[0].config.model_ast == "model_cond6"
    assert agent2._states_by_name["main"].conditions[0].action == ast.ActionList(
        [
            ast.SendAction(to="agent_1", next_state="main"),
        ]
    )
    assert agent2._states_by_name["main"].conditions[1].config.model_ast == "model_default2"
    assert agent2._states_by_name["main"].conditions[1].action == ast.ActionList(
        [
            ast.MessageAction(text="x", role="user"),
        ]
    )


def test_load_assets_prompt() -> None:
    """Test that the default prompt.yaml can be loaded."""

    yaml_data = config.state_machine.yaml_path.read_text()
    dsm = parsing.loads_from_yaml(yaml_data)
    assert isinstance(dsm, ast.StateMachine)


# -*- mode: python; indent-tabs-mode: nil; tab-width: 4 -*-
