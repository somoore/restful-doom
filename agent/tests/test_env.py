import asyncio
from types import SimpleNamespace

from restfuldoom_agent.client import EpisodeReset
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


def _state(*, tick=1, kills=0, enemy=False, combat=False):
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
            distance_fp=256 * 65536,
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
