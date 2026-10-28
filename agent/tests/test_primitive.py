"""Proof-of-life tests for the primitive per-tic action mode."""

import asyncio
from types import SimpleNamespace

import pytest

from restfuldoom_agent.brain import BT_ATTACK, BT_USE
from restfuldoom_agent.env import (
    ACTION_SCHEMA,
    PRIMITIVE_ACTIONS,
    DoomAgentEnv,
    DoomEnvConfig,
    EnvStep,
    PrimitiveController,
    SkillController,
)
from restfuldoom_agent.ppo_agent import evaluate
from restfuldoom_agent.ppo_eval import evaluate_heuristic_policy

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


def test_heuristic_baseline_rejects_primitive_action_space():
    args = SimpleNamespace(
        primitive_actions=True,
        eval_candidate_only=False,
        eval_baseline="heuristic",
    )
    with pytest.raises(ValueError, match="--eval-baseline random"):
        asyncio.run(evaluate(args))

    with pytest.raises(ValueError, match="primitive_actions"):
        asyncio.run(evaluate_heuristic_policy(DoomEnvConfig(primitive_actions=True)))


def test_allowed_actions_filter_uses_primitive_action_names():
    env = DoomAgentEnv(DoomEnvConfig(primitive_actions=True, allowed_skills=("fire",)))
    mask = env.action_mask()
    assert len(mask) == len(PRIMITIVE_ACTIONS)
    assert [PRIMITIVE_ACTIONS[i] for i, allowed in enumerate(mask) if allowed] == ["fire"]

    combo = DoomAgentEnv(
        DoomEnvConfig(
            primitive_actions=True,
            allowed_skills=("forward_fire", "use"),
            strict_allowed_skills=True,
        )
    )
    combo_mask = combo.action_mask()
    assert [PRIMITIVE_ACTIONS[i] for i, allowed in enumerate(combo_mask) if allowed] == [
        "use",
        "forward_fire",
    ]


def test_allowed_actions_must_exist_in_active_action_space():
    with pytest.raises(ValueError, match="primitive action space"):
        DoomAgentEnv(DoomEnvConfig(primitive_actions=True, allowed_skills=("engage",)))
    with pytest.raises(ValueError, match="skill action space"):
        DoomAgentEnv(DoomEnvConfig(allowed_skills=("forward_fire",)))


def _save_checkpoint(tmp_path, name, *, action_dim, action_space=None, drop_space=False):
    torch = pytest.importorskip("torch")
    from restfuldoom_agent.ppo import PPOConfig, PPOTrainer

    kwargs = {} if action_space is None else {"action_space": action_space}
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=action_dim,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8),
        **kwargs,
    )
    path = trainer.save_checkpoint(tmp_path / name)
    if drop_space:
        # Checkpoints written before the action space was recorded.
        payload = torch.load(path, map_location="cpu")
        payload.pop("action_space")
        payload["action_schema"] = ACTION_SCHEMA
        torch.save(payload, path)
    return path


def test_checkpoint_records_and_enforces_action_space(tmp_path):
    from restfuldoom_agent.ppo import PPOTrainer

    skill_dim = len(ACTION_SCHEMA["actions"])
    primitive_dim = len(PRIMITIVE_ACTIONS)

    skill = _save_checkpoint(tmp_path, "skill.pt", action_dim=skill_dim)
    with pytest.raises(ValueError, match="'skill' action space"):
        PPOTrainer.load_checkpoint(
            skill,
            target_obs_dim=2,
            target_action_dim=primitive_dim,
            target_action_space="primitive",
        )
    assert PPOTrainer.load_checkpoint(skill, target_action_space="skill").action_space == "skill"

    primitive = _save_checkpoint(
        tmp_path, "primitive.pt", action_dim=primitive_dim, action_space="primitive"
    )
    loaded = PPOTrainer.load_checkpoint(
        primitive,
        target_obs_dim=2,
        target_action_dim=primitive_dim,
        target_action_space="primitive",
    )
    assert loaded.action_space == "primitive"
    assert loaded.resume_migration is None
    with pytest.raises(ValueError, match="'primitive' action space"):
        PPOTrainer.load_checkpoint(primitive, target_action_space="skill")


def test_legacy_checkpoints_infer_action_space(tmp_path):
    from restfuldoom_agent.ppo import PPOTrainer

    skill_dim = len(ACTION_SCHEMA["actions"])
    primitive_dim = len(PRIMITIVE_ACTIONS)

    legacy_skill = _save_checkpoint(
        tmp_path, "legacy-skill.pt", action_dim=skill_dim, drop_space=True
    )
    with pytest.raises(ValueError, match="'skill' action space"):
        PPOTrainer.load_checkpoint(
            legacy_skill,
            target_obs_dim=2,
            target_action_dim=primitive_dim,
            target_action_space="primitive",
        )

    # Primitive runs from before the field existed saved the skill schema next
    # to a wider primitive head; that shape identifies them.
    legacy_primitive = _save_checkpoint(
        tmp_path,
        "legacy-primitive.pt",
        action_dim=primitive_dim,
        action_space="primitive",
        drop_space=True,
    )
    loaded = PPOTrainer.load_checkpoint(
        legacy_primitive,
        target_obs_dim=2,
        target_action_dim=primitive_dim,
        target_action_space="primitive",
    )
    assert loaded.action_space == "primitive"
    with pytest.raises(ValueError, match="'primitive' action space"):
        PPOTrainer.load_checkpoint(legacy_primitive, target_action_space="skill")
