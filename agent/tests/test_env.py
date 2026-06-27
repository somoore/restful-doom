import asyncio
from types import SimpleNamespace

import pytest

from restfuldoom_agent.client import EpisodeReset
from restfuldoom_agent.client import agent_pb2
from restfuldoom_agent.env import DoomAgentEnv, DoomEnvConfig, SKILL_ACTIONS, SkillController


def test_skill_controller_encodes_observation_and_executes_each_skill(tmp_path):
    controller = SkillController()
    state = _state(enemy=True, combat=True)

    obs = controller.observation(state)

    assert len(obs) > 10
    for index, skill in enumerate(SKILL_ACTIONS):
        action, decision = controller.action_for(index, state)
        assert action is not None
        assert decision["ppo_skill"] == skill
        assert decision["ppo_action_index"] == index


def test_doom_agent_env_reset_step_with_fake_client():
    first = _state(tick=1, kills=0)
    second = _state(tick=2, kills=1)
    client = _FakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=2, goal_preset="combat"),
        client=client,
        controller=SkillController(),
    )

    async def run():
        obs = await env.reset(seed=99)
        step = await env.step(1)
        await env.close()
        return obs, step

    obs, step = asyncio.run(run())

    assert len(obs) == len(step.observation)
    assert client.reset_requests == [{"skill": 2, "episode": 1, "map": 1, "seed": 99}]
    assert step.reward > 0
    assert step.info["skill"] == "fire"
    assert not step.done


def test_doom_agent_env_aggregates_macro_action_tics():
    first = _state(tick=1, kills=0)
    second = _state(tick=2, kills=0, enemy=True, enemy_distance=500)
    third = _state(tick=3, kills=0, enemy=True, enemy_distance=475)
    fourth = _state(tick=4, kills=0, enemy=True, enemy_distance=450)
    client = _DurationAwareFakeClient([first, second, third, fourth])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=10, goal_preset="combat", max_action_tics=4),
        client=client,
        controller=_FixedDurationController(duration_tics=3),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(0)
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["macro_tics"] == 3
    assert step.info["transition"]["enemy_distance_delta"] == 50
    assert step.reward > 0


def test_doom_agent_env_rewards_fire_on_shootable_target():
    first = _state(tick=1, enemy=True, combat=True)
    second = _state(tick=2, enemy=True, combat=True)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=10, goal_preset="combat"),
        client=client,
        controller=_FixedDurationController(duration_tics=1),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(1)
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["skill"] == "fire"
    assert step.info["had_shootable_target"]
    assert step.info["action_reward"] == pytest.approx(0.5)
    assert step.reward == pytest.approx(0.5)


def test_doom_agent_env_penalizes_missed_shootable_target():
    first = _state(tick=1, enemy=True, combat=True)
    second = _state(tick=2, enemy=True, combat=True)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=10, goal_preset="combat"),
        client=client,
        controller=_FixedDurationController(duration_tics=1),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(0)
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["skill"] == "engage"
    assert step.info["had_shootable_target"]
    assert step.info["action_reward"] == pytest.approx(-0.05)
    assert step.reward == pytest.approx(-0.05)


def test_doom_agent_env_reset_can_warmup_until_shootable_target():
    first = _state(tick=1, enemy=True, combat=False)
    second = _state(tick=2, enemy=True, combat=True)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="combat",
            reset_warmup_steps=4,
            reset_warmup_until_shootable=True,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        obs = await env.reset(seed=5)
        await env.close()
        return obs, env._current_state

    obs, current_state = asyncio.run(run())

    assert len(obs) > 10
    assert current_state.tick == 2
    assert current_state.combat.has_shootable_target


def test_doom_agent_env_reset_warmup_respects_tic_limit():
    first = _state(tick=1, enemy=False, combat=False)
    second = _state(tick=2, enemy=False, combat=False)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="combat",
            reset_warmup_steps=4,
            reset_warmup_max_tics=1,
            reset_warmup_until_shootable=True,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=5)
        await env.close()
        return env._last_reset_warmup

    warmup = asyncio.run(run())

    assert warmup["tics"] == 1
    assert warmup["stop_reason"] == "tic_limit"


class _FakeClient:
    def __init__(self, states):
        self.states = list(states)
        self.actions = []
        self.reset_requests = []

    async def reset_episode(self, *, skill, episode, map, seed, run_id):
        self.reset_requests.append(
            {"skill": skill, "episode": episode, "map": map, "seed": seed}
        )
        return EpisodeReset(
            accepted=True,
            message="queued",
            skill=skill,
            episode=episode,
            map=map,
            seed=seed,
            seed_applied=False,
        )

    async def session(self, actions):
        async for action in actions:
            self.actions.append(action)
            if not self.states:
                return
            yield self.states.pop(0)


class _DurationAwareFakeClient(_FakeClient):
    async def session(self, actions):
        async for action in actions:
            self.actions.append(action)
            count = max(1, int(getattr(action, "duration_tics", 0) or 1))
            for _ in range(count):
                if not self.states:
                    return
                yield self.states.pop(0)


class _FixedDurationController:
    def __init__(self, duration_tics):
        self.duration_tics = duration_tics

    def observation(self, _state):
        return [0.0, 1.0]

    def action_for(self, action_index, state):
        return (
            agent_pb2.PlayerAction(duration_tics=self.duration_tics),
            {
                "ppo_skill": "engage",
                "ppo_action_index": action_index,
                "tick": state.tick,
            },
        )


def _state(*, tick=1, kills=0, enemy=False, combat=False, enemy_distance=256):
    position = SimpleNamespace(x_fp=0, y_fp=0, z_fp=0)
    obj = SimpleNamespace(
        id=1,
        position=position,
        angle_degrees=0,
        height_fp=0,
        health=100,
        type_id=0,
        internal_type=0,
        flags=0,
        attacking_id=0,
        distance_fp=0,
    )
    player = SimpleNamespace(
        object=obj,
        health=100,
        armor=0,
        kills=kills,
        items=0,
        secrets=0,
        ready_weapon=1,
        ammo=SimpleNamespace(bullets=50, shells=0, cells=0, rockets=0),
        key_cards=0,
        cheat_flags=0,
        active=True,
    )
    enemies = []
    if enemy:
        enemy_position = SimpleNamespace(x_fp=256 * 65536, y_fp=0, z_fp=0)
        enemy_obj = SimpleNamespace(
            id=7,
            position=enemy_position,
            angle_degrees=0,
            height_fp=0,
            health=20,
            type_id=0,
            internal_type=0,
            flags=0,
            attacking_id=0,
            distance_fp=enemy_distance * 65536,
        )
        enemies.append(SimpleNamespace(object=enemy_obj, line_of_sight=True, target_id=0))
    navigation = SimpleNamespace(
        forward_open=True,
        back_open=True,
        left_open=True,
        right_open=True,
        use_line_ahead=False,
        front_blocking_line_special=0,
        front_block_distance_fp=96 * 65536,
        probe_distance_fp=96 * 65536,
        direction_probes=[],
        use_lines=[],
    )
    combat_probe = SimpleNamespace(
        has_shootable_target=combat,
        target_id=7 if combat else 0,
        target_health=20 if combat else 0,
        target_distance_fp=256 * 65536 if combat else 0,
        aim_slope_fp=0,
        range_fp=2048 * 65536,
        target_is_enemy=combat,
    )
    return SimpleNamespace(
        tick=tick,
        player=player,
        enemies=enemies,
        objects=[],
        level=SimpleNamespace(
            episode=1,
            map=1,
            skill=2,
            level_time=0,
            total_kills=1,
            total_items=0,
            total_secrets=0,
            gamestate=0,
        ),
        has_delta_state=False,
        navigation=navigation,
        combat=combat_probe,
    )
