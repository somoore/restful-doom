"""Proof-of-life tests for the primitive per-tic action mode."""

import asyncio

from restfuldoom_agent.brain import BT_ATTACK, BT_USE
from restfuldoom_agent.env import (
    PRIMITIVE_ACTIONS,
    DoomAgentEnv,
    DoomEnvConfig,
    EnvStep,
    PrimitiveController,
    SkillController,
)

from tests.test_env import _FakeClient, _state


def test_primitive_controller_action_space_and_ticcmd_mapping():
    controller = PrimitiveController()
    state = _state(enemy=True, combat=True)

    assert len(controller.actions) == len(PRIMITIVE_ACTIONS)
    assert controller.action_mask(state) == [True] * len(PRIMITIVE_ACTIONS)

    by_name = {name: index for index, name in enumerate(PRIMITIVE_ACTIONS)}

    noop_action, noop_decision = controller.action_for(by_name["noop"], state)
    assert noop_decision["primitive"] == "noop"
    assert noop_decision["ppo_action_index"] == by_name["noop"]
    assert noop_action.raw.forward_move == 0
    assert noop_action.raw.side_move == 0
    assert noop_action.raw.angle_turn == 0
    assert noop_action.raw.buttons == 0
    assert noop_action.duration_tics == 1

    forward_action, _ = controller.action_for(by_name["forward"], state)
    assert forward_action.raw.forward_move > 0

    backward_action, _ = controller.action_for(by_name["backward"], state)
    assert backward_action.raw.forward_move < 0

    turn_left_action, _ = controller.action_for(by_name["turn_left"], state)
    assert turn_left_action.raw.angle_turn > 0

    turn_right_action, _ = controller.action_for(by_name["turn_right"], state)
    assert turn_right_action.raw.angle_turn < 0

    strafe_left_action, _ = controller.action_for(by_name["strafe_left"], state)
    assert strafe_left_action.raw.side_move < 0

    strafe_right_action, _ = controller.action_for(by_name["strafe_right"], state)
    assert strafe_right_action.raw.side_move > 0

    fire_action, _ = controller.action_for(by_name["fire"], state)
    assert fire_action.raw.buttons & BT_ATTACK

    use_action, _ = controller.action_for(by_name["use"], state)
    assert use_action.raw.buttons & BT_USE

    combo_action, _ = controller.action_for(by_name["forward_fire"], state)
    assert combo_action.raw.forward_move > 0
    assert combo_action.raw.buttons & BT_ATTACK

    combo_turn, _ = controller.action_for(by_name["forward_turn_left"], state)
    assert combo_turn.raw.forward_move > 0
    assert combo_turn.raw.angle_turn > 0


def test_primitive_observation_matches_skill_observation_length():
    state = _state(enemy=True, combat=True, route=True)
    skill_obs = SkillController().observation(state)
    primitive_obs = PrimitiveController().observation(state)
    assert len(primitive_obs) == len(skill_obs)


def test_env_primitive_mode_uses_primitive_action_space():
    env = DoomAgentEnv(DoomEnvConfig(primitive_actions=True))
    assert isinstance(env.controller, PrimitiveController)
    assert len(env.controller.actions) == len(PRIMITIVE_ACTIONS)
    assert len(env.action_mask()) == len(PRIMITIVE_ACTIONS)

    default_env = DoomAgentEnv(DoomEnvConfig())
    assert isinstance(default_env.controller, SkillController)


def test_env_primitive_mode_step_runs_reward_path():
    first = _state(tick=1, kills=0, enemy=True, combat=True, route=True)
    second = _state(tick=2, kills=1, route=True, x_units=64)
    client = _FakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(primitive_actions=True, max_steps=2, goal_preset="combat"),
        client=client,
    )

    async def run():
        await env.reset(seed=1)
        step = await env.step(PRIMITIVE_ACTIONS.index("forward"))
        await env.close()
        return step

    step = asyncio.run(run())
    assert isinstance(step, EnvStep)
    assert isinstance(step.reward, float)
    assert step.info["skill"] == "forward"
    assert step.info["action_index"] == PRIMITIVE_ACTIONS.index("forward")
    assert step.info["decision"]["primitive"] == "forward"
