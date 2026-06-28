import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from restfuldoom_agent.client import EpisodeReset
from restfuldoom_agent.client import agent_pb2
from restfuldoom_agent.brain import (
    AgentMemory,
    BT_USE,
    LINE_ATTEMPT_STALL_TICS,
    _NON_LOCOMOTION_SKILLS,
    extract_features,
)
from restfuldoom_agent.env import (
    DoomAgentEnv,
    DoomEnvConfig,
    LOW_HEALTH_RETREAT_STREAK_LIMIT,
    SKILL_ACTIONS,
    SkillController,
)
from restfuldoom_agent.schemas import OBSERVATION_SCHEMA


def test_skill_controller_encodes_observation_and_executes_each_skill(tmp_path):
    controller = SkillController()
    state = _state(enemy=True, combat=True)

    obs = controller.observation(state)

    assert len(obs) == len(OBSERVATION_SCHEMA["feature_names"])
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
    assert len(obs) == len(OBSERVATION_SCHEMA["feature_names"])
    assert client.reset_requests == [
        {"skill": 2, "episode": 1, "map": 1, "seed": 99, "start": None}
    ]
    assert step.reward > 0
    assert step.info["skill"] == "fire"
    assert step.info["decision_cycle"]["schema"] == "restfuldoom.decision_cycle.v1"
    assert step.info["decision_cycle"]["observation_schema"] == "restfuldoom.observation.v1"
    assert step.info["decision_cycle"]["action_schema"] == "restfuldoom.skill_action.v1"
    assert step.info["decision_cycle"]["memory_contract"] == (
        "restfuldoom.agent_memory_contract.v1"
    )
    assert step.info["decision_cycle"]["input_tick"] == 1
    assert step.info["decision_cycle"]["output_tick"] == 2
    assert not step.done


def test_doom_agent_env_terminates_on_required_kills_after_reset():
    first = _state(tick=1, kills=0)
    second = _state(tick=2, kills=1)
    client = _FakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="combat",
            required_kills=1,
            kill_goal_bonus=10.0,
            terminate_on_required_kills=True,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=99)
        step = await env.step(SKILL_ACTIONS.index("fire"))
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.done
    assert step.info["done_reason"] == "required_kills"
    assert step.info["transition"]["kill_delta"] == 1
    assert step.reward >= 10.0


def test_skill_controller_observation_includes_previous_action_history():
    controller = SkillController()
    state = _state(enemy=True, combat=True)

    initial = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))
    controller.record_action_history(action_index=1, had_shootable_target=True)
    after_fire = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert initial["prev_skill_fire"] == 0.0
    assert initial["prev_had_shootable_target"] == 0.0
    assert after_fire["prev_skill_fire"] == 1.0
    assert after_fire["prev_had_shootable_target"] == 1.0
    assert after_fire["same_skill_streak_norm"] > 0.0


def test_skill_controller_reset_clears_policy_episode_context():
    controller = SkillController()
    policy = controller.policy
    controller.last_decision = {"skill": "old"}
    controller._previous_action_index = SKILL_ACTIONS.index("seek_enemy")
    controller._same_skill_streak = 7
    controller._recent_visible_enemy_flags.extend([True, False])
    controller._recent_route_progress_units.extend([32.0])
    controller._recent_contact_use_line = {"tick": 10}
    controller._recent_visible_contact = {"tick": 11}
    policy.last_decision = {"skill": "stale"}
    policy._last_position = (1.0, 2.0)
    policy._last_progress_tick = 99
    policy._last_shot_tick = 88
    policy._last_use_tick = 77
    policy._last_stuck_phase = 3
    policy._explore_bias = -1
    policy._blocked_enemy_cells["1:2:3"] = 100
    policy._blocked_use_lines["1:2:4"] = 100
    policy._line_attempts["1:2:4"] = {"count": 2}
    policy._exit_push_attempts["1:2:5"] = {"count": 1}
    policy._episode_cell_visits["1:2"] = 12
    policy._start_kills = 4
    policy._last_visible_enemy_tick = 55
    policy._last_visible_enemy_id = 6
    policy._last_contact_ray = {"tick": 55, "enemy_id": 6, "ray_offset": -45}
    policy._hazard_escape = {"started_tick": 66}

    controller.reset_episode_context()

    assert controller.last_decision == {}
    assert controller._previous_action_index is None
    assert controller._same_skill_streak == 0
    assert controller._recent_visible_enemy_flags == []
    assert controller._recent_route_progress_units == []
    assert controller._recent_contact_use_line is None
    assert controller._recent_visible_contact is None
    assert policy.last_decision == {}
    assert policy._last_position is None
    assert policy._last_progress_tick == 0
    assert policy._last_shot_tick == -9999
    assert policy._last_use_tick == -9999
    assert policy._last_stuck_phase == -1
    assert policy._explore_bias == 1
    assert policy._blocked_enemy_cells == {}
    assert policy._blocked_use_lines == {}
    assert policy._line_attempts == {}
    assert policy._exit_push_attempts == {}
    assert policy._episode_cell_visits == {}
    assert policy._start_kills is None
    assert policy._last_visible_enemy_tick == -9999
    assert policy._last_visible_enemy_id is None
    assert policy._last_contact_ray is None
    assert policy._hazard_escape is None


def test_skill_controller_observation_includes_route_outcome_history():
    controller = SkillController()
    state = _state(route=True)
    route_index = SKILL_ACTIONS.index("route_progression")

    controller.record_action_history(
        action_index=route_index,
        had_shootable_target=False,
        route_outcome={
            "attempted": True,
            "progress_units": 128.0,
            "reached": True,
            "failed": False,
        },
    )
    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert features["prev_skill_route_progression"] == 1.0
    assert features["prev_route_progression"] == 1.0
    assert features["prev_route_progress_norm"] == pytest.approx(0.5)
    assert features["route_waypoint_reached_recently"] == 1.0
    assert features["route_waypoint_failed_recently"] == 0.0
    assert features["failed_route_attempt_count_norm"] == 0.0


def test_skill_controller_observation_includes_temporal_context():
    controller = SkillController()
    first = _state(enemy=True, route=True, enemy_distance=512, x_units=0)
    second = _state(enemy=True, route=True, enemy_distance=384, x_units=128)

    initial = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(first)))
    after_move = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(second)))

    assert initial["delta_x_norm"] == 0.0
    assert initial["movement_distance_norm"] == 0.0
    assert initial["enemy_distance_delta_norm"] == 0.0
    assert after_move["delta_x_norm"] == pytest.approx(0.25)
    assert after_move["movement_distance_norm"] == pytest.approx(0.25)
    assert after_move["enemy_distance_delta_norm"] == pytest.approx(0.25)
    assert after_move["route_distance_delta_norm"] == pytest.approx(0.25)
    assert after_move["cell_changed_recently"] == 1.0
    assert after_move["visible_enemy_seen_recently"] == 1.0


def test_skill_controller_observation_tracks_recent_route_failures():
    controller = SkillController()
    route_index = SKILL_ACTIONS.index("route_progression")
    state = _state(route=True)

    controller.record_action_history(
        action_index=route_index,
        had_shootable_target=False,
        route_outcome={
            "attempted": True,
            "progress_units": -16.0,
            "reached": False,
            "failed": True,
        },
    )
    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert features["recent_route_progress_norm"] == pytest.approx(-16.0 / 512.0)
    assert features["recent_route_failure_ratio"] == 1.0


def test_skill_controller_observation_includes_current_contact_use_line_context():
    controller = SkillController()
    state = _state(
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=180,
    )

    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert features["recent_contact_active"] == 1.0
    assert features["contact_use_line_active"] == 1.0
    assert features["contact_use_line_distance_norm"] == pytest.approx(180.0 / 1400.0)
    assert features["contact_use_line_angle_cos"] == pytest.approx(1.0)
    assert features["contact_use_line_close"] == 1.0
    assert features["contact_use_line_followthrough_active"] == 0.0
    assert features["contact_use_line_age_norm"] == 0.0


def test_skill_controller_observation_includes_visible_contact_geometry():
    controller = SkillController()
    visible = _state(enemy=True, combat=False, enemy_distance=1200)
    shootable = _state(enemy=True, combat=True, enemy_distance=256)

    visible_features = dict(
        zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(visible))
    )
    shootable_features = dict(
        zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(shootable))
    )

    assert visible_features["visible_contact_active"] == 1.0
    assert visible_features["visible_contact_shootable"] == 0.0
    assert visible_features["visible_contact_needs_closure"] == 1.0
    assert visible_features["visible_contact_distance_norm"] == pytest.approx(0.5)
    assert visible_features["visible_contact_angle_cos"] == pytest.approx(1.0)
    assert visible_features["visible_contact_aligned"] == 1.0
    assert visible_features["visible_contact_close"] == 0.0
    assert shootable_features["visible_contact_active"] == 1.0
    assert shootable_features["visible_contact_shootable"] == 1.0
    assert shootable_features["visible_contact_needs_closure"] == 0.0
    assert shootable_features["visible_contact_close"] == 1.0


def test_skill_controller_observation_includes_remembered_contact_use_line_context():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=80,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
        contact_use_distance_units=640,
    )

    controller.action_mask(first)
    controller.record_action_history(
        action_index=SKILL_ACTIONS.index("open_use_line"),
        had_shootable_target=False,
    )
    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(second)))

    assert features["recent_contact_active"] == 1.0
    assert features["contact_use_line_active"] == 1.0
    assert features["contact_use_line_distance_norm"] == pytest.approx(640.0 / 1400.0)
    assert features["contact_use_line_close"] == 0.0
    assert features["contact_use_line_followthrough_active"] == 1.0
    assert features["contact_use_line_age_norm"] == pytest.approx(75.0 / 420.0)


def test_skill_controller_action_mask_uses_affordances():
    controller = SkillController()
    combat_state = _state(enemy=True, combat=True)
    visible_not_shootable_state = _state(enemy=True, combat=False)
    visible_route_state = _state(enemy=True, combat=False, route=True)
    visible_use_state = _state(enemy=True, combat=False, contact_use=True)
    quiet_state = _state(enemy=False, combat=False)

    combat_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(combat_state)))
    visible_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(visible_not_shootable_state)))
    visible_route_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(visible_route_state)))
    visible_use_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(visible_use_state)))
    quiet_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(quiet_state)))

    assert combat_mask["fire"]
    assert not combat_mask["engage"]
    assert not combat_mask["retreat"]
    assert not combat_mask["seek_enemy"]
    assert not combat_mask["open_use_line"]
    assert not combat_mask["route_progression"]
    assert not combat_mask["press_exit"]
    assert not visible_mask["fire"]
    assert not visible_mask["engage"]
    assert visible_mask["close_visible_contact"]
    assert visible_mask["seek_enemy"]
    assert not visible_route_mask["route_progression"]
    assert visible_use_mask["open_use_line"]
    assert quiet_mask["route_progression"]


def test_skill_controller_fire_masks_out_recent_contact_closure():
    controller = SkillController()
    first_visible = _state(tick=5, enemy=True, combat=False)
    shootable_after_contact = _state(tick=20, enemy=False, combat=True)

    controller.action_mask(first_visible)
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(shootable_after_contact)))

    assert mask["fire"]
    assert not mask["close_visible_contact"]
    assert not mask["seek_enemy"]
    assert not mask["open_use_line"]
    assert not mask["route_progression"]


def test_skill_controller_low_health_contact_forces_retreat():
    controller = SkillController()
    combat_state = _state(enemy=True, combat=True, health=30)
    visible_state = _state(enemy=True, combat=False, health=30)

    combat_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(combat_state)))
    visible_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(visible_state)))

    assert combat_mask["retreat"]
    assert not combat_mask["fire"]
    assert not combat_mask["close_visible_contact"]
    assert visible_mask["retreat"]
    assert not visible_mask["seek_enemy"]
    assert not visible_mask["close_visible_contact"]


def test_skill_controller_low_health_no_visible_contact_breaks_retreat_loop():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False)
    low_health_lost_contact = _state(
        tick=40,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        health=30,
    )

    controller.action_mask(first)
    initial_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(low_health_lost_contact)))
    controller._previous_action_index = SKILL_ACTIONS.index("retreat")
    controller._same_skill_streak = LOW_HEALTH_RETREAT_STREAK_LIMIT
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(low_health_lost_contact)))

    assert initial_mask["retreat"]
    assert not initial_mask["close_visible_contact"]
    assert not mask["retreat"]
    assert mask["close_visible_contact"]
    assert not mask["route_progression"]


def test_skill_controller_suppresses_blind_seek_before_episode_contact(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 10,
            "last_position": [512.0, 0.0],
            "last_distance": 512.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    quiet_spawn = _state(tick=10, enemy=False, combat=False)
    visible_contact = _state(tick=11, enemy=True, combat=False)
    lost_contact = _state(tick=20, enemy=True, enemy_line_of_sight=False, combat=False)
    expired_contact = _state(tick=500, enemy=True, enemy_line_of_sight=False, combat=False)

    quiet_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(quiet_spawn)))
    quiet_heuristic = SKILL_ACTIONS[controller.heuristic_action_index(quiet_spawn)]
    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    lost_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))
    lost_heuristic = SKILL_ACTIONS[controller.heuristic_action_index(lost_contact)]
    for _ in range(16):
        controller.record_action_history(
            action_index=SKILL_ACTIONS.index("close_visible_contact"),
            had_shootable_target=False,
        )
    recovered_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))
    recovered_heuristic = SKILL_ACTIONS[controller.heuristic_action_index(lost_contact)]
    expired_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(expired_contact)))

    assert not quiet_mask["seek_enemy"]
    assert quiet_mask["route_progression"]
    assert quiet_heuristic == "route_progression"
    assert lost_mask["close_visible_contact"]
    assert not lost_mask["seek_enemy"]
    assert lost_heuristic == "route_progression"
    assert not recovered_mask["seek_enemy"]
    assert recovered_heuristic == "route_progression"
    assert not expired_mask["seek_enemy"]
    assert expired_mask["route_progression"]


def test_doom_agent_env_allowed_skill_filter_narrows_action_mask():
    env = DoomAgentEnv(
        DoomEnvConfig(allowed_skills=("close_visible_contact", "fire")),
        controller=SkillController(),
    )

    env._current_state = _state(enemy=True, combat=False)
    visible_mask = dict(zip(SKILL_ACTIONS, env.action_mask()))
    env._current_state = _state(enemy=True, combat=True)
    combat_mask = dict(zip(SKILL_ACTIONS, env.action_mask()))
    env._current_state = _state(enemy=False, combat=False)
    quiet_mask = dict(zip(SKILL_ACTIONS, env.action_mask()))

    assert visible_mask["close_visible_contact"]
    assert not visible_mask["seek_enemy"]
    assert not visible_mask["fire"]
    assert combat_mask["fire"]
    assert not combat_mask["close_visible_contact"]
    assert quiet_mask["route_progression"]
    assert env._last_action_mask_filter["fallback_skill"] == "unfiltered_mask"


def test_doom_agent_env_strict_allowed_skill_filter_stays_inside_allowlist():
    env = DoomAgentEnv(
        DoomEnvConfig(
            allowed_skills=("close_visible_contact", "fire"),
            strict_allowed_skills=True,
        ),
        controller=SkillController(),
    )

    env._current_state = _state(enemy=False, combat=False)
    quiet_mask = dict(zip(SKILL_ACTIONS, env.action_mask()))

    assert quiet_mask["close_visible_contact"]
    assert not quiet_mask["route_progression"]
    assert not quiet_mask["recover_stuck"]
    assert env._last_action_mask_filter["fallback_applied"]
    assert env._last_action_mask_filter["fallback_skill"] == "close_visible_contact"


def test_skill_controller_contact_actions_use_visible_enemy_and_route_waypoint():
    controller = SkillController()
    state = _state(enemy=True, combat=False, enemy_distance=640, route=True, contact_use=True)

    _seek_action, seek_decision = controller.action_for(SKILL_ACTIONS.index("seek_enemy"), state)
    _route_action, route_decision = controller.action_for(
        SKILL_ACTIONS.index("route_progression"),
        state,
    )
    _use_action, use_decision = controller.action_for(
        SKILL_ACTIONS.index("open_use_line"),
        state,
    )

    assert seek_decision["skill"] in {"ppo_seek_visible_contact", "ppo_seek_visible_enemy"}
    assert seek_decision["enemy"]["id"] == 7
    assert route_decision["ppo_skill"] == "route_progression"
    assert route_decision["skill"] in {
        "approach_progression_line",
        "cross_progression_line",
        "turn_to_progression_line",
        "route_to_progression_line",
        "use_progression_line",
    }
    assert use_decision["ppo_skill"] == "open_use_line"
    assert use_decision["use_line"]["line_id"] == 151


def test_skill_controller_far_visible_contact_uses_close_option_before_use_line():
    controller = SkillController()
    state = _state(enemy=True, combat=False, enemy_distance=1800, contact_use=True)

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("open_use_line"), state)

    assert mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert decision["skill"] == "ppo_close_visible_contact"
    assert decision["ppo_skill"] == "open_use_line"


def test_skill_controller_close_visible_contact_prefers_open_side_ray():
    controller = SkillController()
    state = _state(
        enemy=True,
        combat=False,
        enemy_distance=1800,
        direction_probes=[
            {"angle_offset_degrees": 0, "open": False, "block_distance": 96},
            {"angle_offset_degrees": -90, "open": True, "block_distance": 512},
            {"angle_offset_degrees": 90, "open": True, "block_distance": 512},
        ],
    )
    state.navigation.forward_open = False

    _close_action, close_decision = controller.action_for(
        SKILL_ACTIONS.index("close_visible_contact"),
        state,
    )
    _seek_action, seek_decision = controller.action_for(
        SKILL_ACTIONS.index("seek_enemy"),
        state,
    )

    assert close_decision["skill"] == "ppo_close_visible_contact"
    assert close_decision["direction_probe"]["open"] is True
    assert close_decision["direction_probe"]["angle_offset_degrees"] == -90
    assert seek_decision["skill"] == "ppo_seek_visible_contact"
    assert seek_decision["direction_probe"]["open"] is False
    assert seek_decision["direction_probe"]["angle_offset_degrees"] == 0


def test_skill_controller_close_visible_contact_approaches_contact_line():
    controller = SkillController()
    state = _state(
        enemy=True,
        combat=False,
        enemy_distance=1800,
        contact_use=True,
        contact_use_distance_units=640,
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(
        SKILL_ACTIONS.index("close_visible_contact"),
        state,
    )

    assert mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert decision["ppo_skill"] == "close_visible_contact"
    assert decision["skill"] == "contact_approach_use_line"
    assert decision["use_line"]["line_id"] == 151


def test_skill_controller_close_visible_contact_continues_contact_line_after_los_drops():
    controller = SkillController()
    first = _state(
        tick=5,
        enemy=True,
        combat=False,
        enemy_distance=1800,
        contact_use=True,
        contact_use_distance_units=640,
    )
    second = _state(
        tick=24,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        enemy_distance=1800,
        contact_use=True,
        contact_use_distance_units=560,
    )

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), first)
    _action, decision = controller.action_for(
        SKILL_ACTIONS.index("close_visible_contact"),
        second,
    )

    assert decision["ppo_skill"] == "close_visible_contact"
    assert decision["skill"] == "contact_approach_use_line"
    assert decision["use_line"]["line_id"] == 151


def test_skill_controller_close_visible_contact_releases_stalled_contact_line():
    controller = SkillController()
    first = _state(
        tick=5,
        enemy=True,
        combat=False,
        enemy_distance=1800,
        contact_use=True,
        contact_use_distance_units=640,
    )
    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), first)

    last_decision = None
    for tick in range(20, 80, 5):
        stalled = _state(
            tick=tick,
            enemy=True,
            enemy_line_of_sight=False,
            combat=False,
            enemy_distance=1800,
            contact_use=True,
            contact_use_distance_units=16,
        )
        stalled.navigation.use_line_ahead = True
        _action, last_decision = controller.action_for(
            SKILL_ACTIONS.index("close_visible_contact"),
            stalled,
        )

    assert last_decision is not None
    assert last_decision["ppo_skill"] == "close_visible_contact"
    assert last_decision["skill"] != "contact_use_line"


def test_skill_controller_engage_continues_recent_contact_corridor():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, enemy_distance=2200)
    second = _state(
        tick=20,
        enemy=True,
        combat=False,
        enemy_distance=2200,
        enemy_line_of_sight=False,
    )

    _first_action, first_decision = controller.action_for(
        SKILL_ACTIONS.index("engage"),
        first,
    )
    _second_action, second_decision = controller.action_for(
        SKILL_ACTIONS.index("engage"),
        second,
    )

    assert first_decision["skill"] == "close_visible_contact"
    assert second_decision["skill"] == "pursue_last_contact_corridor"
    assert second_decision["ppo_skill"] == "engage"


def test_skill_controller_open_use_line_remembers_contact_line_after_los_drops():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=20,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
    )

    _first_action, first_decision = controller.action_for(
        SKILL_ACTIONS.index("open_use_line"),
        first,
    )
    _second_action, second_decision = controller.action_for(
        SKILL_ACTIONS.index("open_use_line"),
        second,
    )

    assert first_decision["use_line"]["line_id"] == 151
    assert second_decision["use_line"]["line_id"] == 151
    assert second_decision["skill"] != "ppo_use_ahead"


def test_skill_controller_mask_remembers_contact_line_without_open_action():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=80,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
    )

    first_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(first)))
    second_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(second)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("open_use_line"), second)

    assert first_mask["open_use_line"]
    assert not second_mask["open_use_line"]
    assert second_mask["close_visible_contact"]
    assert decision["use_line"]["line_id"] == 151
    assert decision["skill"] != "ppo_use_ahead"


def test_skill_controller_contact_use_line_followthrough_masks_to_open_use_line():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=20,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
    )

    controller.action_mask(first)
    controller.record_action_history(
        action_index=SKILL_ACTIONS.index("open_use_line"),
        had_shootable_target=False,
    )
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(second)))

    assert mask["open_use_line"]
    assert not mask["engage"]
    assert not mask["seek_enemy"]
    assert not mask["route_progression"]


def test_skill_controller_contact_use_line_followthrough_releases_after_streak():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=80,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
        contact_use_distance_units=640,
    )

    controller.action_mask(first)
    for _ in range(16):
        controller.record_action_history(
            action_index=SKILL_ACTIONS.index("open_use_line"),
            had_shootable_target=False,
        )
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(second)))

    assert not mask["open_use_line"]
    assert not mask["engage"]
    assert mask["close_visible_contact"]
    assert not mask["seek_enemy"]


def test_skill_controller_contact_use_line_allows_doorway_approach_range():
    controller = SkillController()
    state = _state(
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=960,
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("open_use_line"), state)

    assert mask["open_use_line"]
    assert decision["use_line"]["line_id"] == 151
    assert decision["skill"] in {
        "contact_approach_use_line",
        "approach_nearby_use_line",
        "turn_to_nearby_use_line",
        "use_nearby_line",
    }


def test_skill_controller_contact_use_line_bypasses_blacklisted_line_attempt():
    controller = SkillController()
    state = _state(enemy=True, combat=False, contact_use=True)
    controller.policy._blocked_use_lines["0:0:151"] = state.tick + 1000

    _action, decision = controller.action_for(SKILL_ACTIONS.index("open_use_line"), state)

    assert decision["use_line"]["line_id"] == 151
    assert decision["skill"] == "contact_approach_use_line"


def test_skill_controller_contact_use_line_presses_use_when_close():
    controller = SkillController()
    state = _state(
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=180,
    )

    action, decision = controller.action_for(SKILL_ACTIONS.index("open_use_line"), state)

    assert decision["skill"] == "contact_use_line"
    assert action.raw.buttons & BT_USE
    assert action.raw.forward_move > 0


def test_skill_controller_recent_contact_mask_suppresses_generic_route():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False, contact_use=True)
    second = _state(
        tick=20,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        contact_use=True,
    )

    controller.action_for(SKILL_ACTIONS.index("open_use_line"), first)
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(second)))

    assert not mask["engage"]
    assert mask["close_visible_contact"]
    assert not mask["seek_enemy"]
    assert not mask["open_use_line"]
    assert not mask["route_progression"]


def test_skill_controller_recent_visible_contact_suppresses_generic_route():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False)
    second = _state(
        tick=20,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
    )
    expired = _state(
        tick=500,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
    )

    controller.action_mask(first)
    contact_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(second)))
    expired_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(expired)))

    assert not contact_mask["engage"]
    assert contact_mask["close_visible_contact"]
    assert not contact_mask["seek_enemy"]
    assert not contact_mask["route_progression"]
    assert expired_mask["route_progression"]
    assert not expired_mask["close_visible_contact"]


def test_skill_controller_recent_contact_route_failures_suppress_progression_line():
    controller = SkillController()
    controller.policy._start_kills = 0
    first = _state(tick=5, kills=1, enemy=True, combat=False)
    route_state = _state(
        tick=20,
        kills=1,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        route=True,
    )
    route_index = SKILL_ACTIONS.index("route_progression")

    controller.action_mask(first)
    before_failures = dict(zip(SKILL_ACTIONS, controller.action_mask(route_state)))
    controller.record_action_history(
        action_index=route_index,
        had_shootable_target=False,
        route_outcome={
            "attempted": True,
            "progress_units": -8.0,
            "reached": False,
            "failed": True,
        },
    )
    after_failure = dict(zip(SKILL_ACTIONS, controller.action_mask(route_state)))
    combat_state = _state(
        tick=22,
        kills=1,
        enemy=True,
        enemy_line_of_sight=True,
        combat=True,
        route=True,
    )
    combat_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(combat_state)))

    assert before_failures["route_progression"]
    assert not after_failure["engage"]
    assert after_failure["close_visible_contact"]
    assert not after_failure["seek_enemy"]
    assert not after_failure["route_progression"]
    assert combat_mask["fire"]
    assert not combat_mask["route_progression"]


def test_exit_switch_decisions_do_not_trigger_stuck_recovery():
    assert "push_exit_switch" in _NON_LOCOMOTION_SKILLS
    assert "press_exit_switch" in _NON_LOCOMOTION_SKILLS
    assert "backtrack_from_exit_switch" in _NON_LOCOMOTION_SKILLS
    assert "turn_to_exit_switch" in _NON_LOCOMOTION_SKILLS


def test_unstick_turn_does_not_reset_recovery_phase():
    controller = SkillController()
    policy = controller.policy
    initial = extract_features(_state(tick=10), controller.memory, controller.params)

    assert not policy._is_stuck(initial)

    stuck_features = extract_features(_state(tick=30), controller.memory, controller.params)
    assert policy._is_stuck(stuck_features)
    _action, decision = policy._recover_from_stuck(stuck_features)

    assert decision["skill"] == "unstick_turn"
    assert decision["stuck_phase"] == 2

    policy.last_decision = decision
    next_features = extract_features(_state(tick=46), controller.memory, controller.params)

    assert policy._is_stuck(next_features)
    assert policy._last_progress_tick == 10
    _next_action, next_decision = policy._recover_from_stuck(next_features)
    assert next_decision["stuck_phase"] == 4


def test_progression_line_attempts_do_not_blacklist_route_target():
    controller = SkillController()
    controller.policy._start_kills = 0
    features = extract_features(
        _state(tick=20, kills=1, route=True),
        controller.memory,
        controller.params,
    )
    line = features.navigation["route_waypoint"]["line"]

    assert controller.policy._record_line_attempt(features, line)
    stalled = replace(features, tick=features.tick + LINE_ATTEMPT_STALL_TICS + 1)

    assert controller.policy._record_line_attempt(stalled, line)
    assert controller.policy._select_progression_line(stalled) is not None


def test_skill_controller_successful_route_outcome_resets_contact_route_backoff():
    controller = SkillController()
    controller.policy._start_kills = 0
    first = _state(tick=5, kills=1, enemy=True, combat=False)
    route_state = _state(
        tick=20,
        kills=1,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        route=True,
    )
    route_index = SKILL_ACTIONS.index("route_progression")

    controller.action_mask(first)
    controller.record_action_history(
        action_index=route_index,
        had_shootable_target=False,
        route_outcome={
            "attempted": True,
            "progress_units": -8.0,
            "reached": False,
            "failed": True,
        },
    )
    suppressed = dict(zip(SKILL_ACTIONS, controller.action_mask(route_state)))
    controller.record_action_history(
        action_index=route_index,
        had_shootable_target=False,
        route_outcome={
            "attempted": True,
            "progress_units": 32.0,
            "reached": False,
            "failed": False,
        },
    )
    restored = dict(zip(SKILL_ACTIONS, controller.action_mask(route_state)))

    assert not suppressed["route_progression"]
    assert restored["route_progression"]


def test_skill_controller_observation_includes_sector_and_route_features():
    controller = SkillController()
    state = _state(hazard=True, route=True)

    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert features["sector_damaging"] == 1.0
    assert features["sector_damage_norm"] == pytest.approx(0.5)
    assert features["sector_exit_damage"] == 0.0
    assert features["route_has_waypoint"] == 1.0
    assert features["route_waypoint_distance_norm"] > 0.0
    assert features["route_waypoint_angle_cos"] == pytest.approx(1.0)
    assert features["route_waypoint_walk_trigger"] == 1.0


def test_skill_controller_observation_includes_topology_frontier_count(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    controller = SkillController(memory=memory)
    state = _state(
        direction_probes=[
            {"angle_offset_degrees": 0, "open": True},
            {"angle_offset_degrees": 90, "open": True},
            {"angle_offset_degrees": -90, "open": True},
        ],
    )

    open_frontier = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))
    memory.data["cells"] = {
        "1:0": {"visits": 3},
        "0:1": {"visits": 3},
        "0:-1": {"visits": 3},
    }
    exhausted = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert open_frontier["topology_frontier_count_norm"] == pytest.approx(1.0)
    assert open_frontier["topology_frontier_active"] == 1.0
    assert open_frontier["topology_frontier_angle_sin"] == pytest.approx(0.0)
    assert open_frontier["topology_frontier_angle_cos"] == pytest.approx(1.0)
    assert open_frontier["topology_exhausted_open_ratio"] == pytest.approx(0.0)
    assert exhausted["topology_frontier_count_norm"] == pytest.approx(0.0)
    assert exhausted["topology_frontier_active"] == 0.0
    assert exhausted["topology_exhausted_open_ratio"] == pytest.approx(1.0)


def test_skill_controller_observation_points_to_least_visited_open_cell(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["cells"] = {
        "1:0": {"visits": 4},
        "0:-1": {"visits": 3},
    }
    controller = SkillController(memory=memory)
    state = _state(
        direction_probes=[
            {"angle_offset_degrees": 0, "open": True},
            {"angle_offset_degrees": 90, "open": True},
            {"angle_offset_degrees": -90, "open": True},
        ],
    )

    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], controller.observation(state)))

    assert features["topology_frontier_active"] == 1.0
    assert features["topology_frontier_angle_sin"] == pytest.approx(1.0)
    assert features["topology_frontier_angle_cos"] == pytest.approx(0.0, abs=1e-6)
    assert features["topology_open_cell_min_visit_norm"] == pytest.approx(0.0)
    assert features["topology_open_cell_mean_visit_norm"] == pytest.approx(7 / 24)
    assert features["topology_exhausted_open_ratio"] == pytest.approx(2 / 3)


def test_doom_agent_env_reset_sends_curriculum_start():
    first = _state(tick=1, enemy=False)
    client = _FakeClient([first])
    env = DoomAgentEnv(
        DoomEnvConfig(
            reset_start_x_fp=1024 * 65536,
            reset_start_y_fp=-512 * 65536,
            reset_start_angle_degrees=90,
            reset_start_face_nearest_enemy=True,
            reset_start_health=95,
            reset_start_ammo_bullets=37,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=99)
        await env.close()

    asyncio.run(run())

    start = client.reset_requests[0]["start"]
    assert start.x_fp == 1024 * 65536
    assert start.y_fp == -512 * 65536
    assert start.angle_degrees == 90
    assert start.face_nearest_enemy
    assert start.health == 95
    assert start.ammo_bullets == 37


def test_doom_agent_env_snapshot_reset_observes_without_episode_reset():
    first = _state(tick=700, level_time=12, enemy=True)
    client = _FakeClient([first])
    env = DoomAgentEnv(
        DoomEnvConfig(
            reset_mode="snapshot",
            snapshot={
                "id": "snap-1",
                "path": "/tmp/snap-1",
                "digest": "sha256:test",
            },
            curriculum_stage={
                "name": "first_contact_snapshot",
                "reset_mode": "snapshot",
                "expected_state": {"level_time": 12},
                "snapshot_restore": {
                    "schema": "restfuldoom.snapshot_restore.v1",
                    "returncode": 0,
                    "elapsed_seconds": 0.1,
                    "restore_command_configured": True,
                },
            },
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        obs = await env.reset(seed=99)
        await env.close()
        return obs, env._last_reset_context

    obs, reset_context = asyncio.run(run())

    assert len(obs) == len(OBSERVATION_SCHEMA["feature_names"])
    assert client.reset_requests == []
    assert reset_context["source"] == "snapshot_restore"
    assert reset_context["skipped_reset_episode"] is True
    assert reset_context["snapshot_id"] == "snap-1"
    assert reset_context["restore"]["returncode"] == 0
    assert reset_context["actual_first_state"]["tick"] == 700
    assert reset_context["actual_first_state"]["level_time"] == 12
    assert reset_context["restored_state_verification"]["valid"] is True
    assert reset_context["restored_state_verification"]["compared_fields"] == [
        "level_time"
    ]


def test_doom_agent_env_snapshot_reset_can_load_server_slot():
    first = _state(tick=800, level_time=14, enemy=True)
    client = _FakeClient([first])
    env = DoomAgentEnv(
        DoomEnvConfig(
            reset_mode="snapshot",
            snapshot={
                "id": "slot-3",
                "ref": "save_slot:3",
                "slot": 3,
                "digest": "engine-save-slot",
            },
            curriculum_stage={
                "name": "first_contact_snapshot",
                "reset_mode": "snapshot",
                "expected_state": {"level_time": 14},
                "snapshot_restore": {
                    "schema": "restfuldoom.snapshot_restore.v1",
                    "api_method": "grpc_load_snapshot",
                    "restore_command_configured": False,
                    "slot": 3,
                    "returncode": 0,
                },
            },
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=99)
        await env.close()
        return env._last_reset_context

    reset_context = asyncio.run(run())

    assert client.reset_requests == []
    assert client.load_snapshot_requests == [{"slot": 3, "run_id": "ppo-1-load-snapshot"}]
    assert reset_context["source"] == "snapshot_restore"
    assert reset_context["restore"]["api_method"] == "grpc_load_snapshot"
    assert reset_context["restore"]["returncode"] == 0
    assert reset_context["actual_first_state"]["tick"] == 800
    assert reset_context["actual_first_state"]["level_time"] == 14
    assert reset_context["restored_state_verification"]["valid"] is True
    assert reset_context["restored_state_verification"]["enabled"] is True


def test_doom_agent_env_snapshot_reset_fails_on_unverified_server_slot():
    first = _state(tick=900, level_time=20, enemy=True)
    client = _FakeClient([first])
    env = DoomAgentEnv(
        DoomEnvConfig(
            reset_mode="snapshot",
            snapshot={
                "id": "slot-3",
                "ref": "save_slot:3",
                "slot": 3,
            },
            curriculum_stage={
                "name": "first_contact_snapshot",
                "reset_mode": "snapshot",
                "expected_state": {"level_time": 14},
            },
            snapshot_verify_tick_tolerance=0,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        try:
            await env.reset(seed=99)
        finally:
            await env.close()

    with pytest.raises(RuntimeError, match="snapshot restored-state verification failed"):
        asyncio.run(run())


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


def test_doom_agent_env_rewards_and_terminates_on_first_visible_enemy():
    first = _state(tick=1, enemy=False, combat=False)
    second = _state(tick=2, enemy=True, combat=False)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="navigation",
            first_visible_bonus=3.0,
            terminate_on_first_visible=True,
        ),
        client=client,
        controller=_FixedDurationController(duration_tics=1),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(0)
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.done
    assert step.info["done_reason"] == "first_visible_enemy"
    assert step.info["had_visible_enemy"]
    assert step.info["first_visible_contact"]
    assert not step.info["first_shootable_contact"]
    assert step.info["contact_reward"] == pytest.approx(3.0)
    assert step.reward == pytest.approx(3.0)


def test_doom_agent_env_rewards_and_terminates_on_first_shootable_target():
    first = _state(tick=1, enemy=True, combat=False)
    second = _state(tick=2, enemy=True, combat=True)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="combat",
            first_shootable_bonus=5.0,
            terminate_on_first_shootable=True,
        ),
        client=client,
        controller=_FixedDurationController(duration_tics=1),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(0)
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.done
    assert step.info["done_reason"] == "first_shootable_target"
    assert step.info["had_visible_enemy"]
    assert step.info["had_shootable_target"]
    assert not step.info["first_visible_contact"]
    assert step.info["first_shootable_contact"]
    assert step.info["contact_reward"] == pytest.approx(5.0)
    assert step.reward == pytest.approx(5.0 - 0.05)


def test_doom_agent_env_rewards_visible_contact_distance_progress():
    first = _state(tick=1, enemy=True, combat=False, enemy_distance=512)
    second = _state(tick=2, enemy=True, combat=False, enemy_distance=384)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="custom",
            max_action_tics=1,
            visible_contact_progress_reward=0.001,
        ),
        client=client,
        controller=_FixedDurationController(duration_tics=1),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(0)
        await env.close()
        return step

    step = asyncio.run(run())

    assert not step.done
    assert step.info["visible_contact_distance_delta"] == pytest.approx(128.0)
    assert step.info["visible_contact_progress_reward"] == pytest.approx(0.128)
    assert step.reward == pytest.approx(0.128)


def test_doom_agent_env_records_route_outcome_and_reward():
    first = _state(tick=1, route=True, x_units=0)
    second = _state(tick=2, route=True, x_units=128)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=10, goal_preset="navigation", max_action_tics=1),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(SKILL_ACTIONS.index("route_progression"))
        await env.close()
        return step

    step = asyncio.run(run())
    features = dict(zip(OBSERVATION_SCHEMA["feature_names"], step.observation))

    assert step.info["route_outcome"]["attempted"]
    assert step.info["route_outcome"]["progress_units"] == pytest.approx(128.0)
    assert step.info["route_outcome"]["failed"] is False
    assert step.info["route_action_reward"] > 0.0
    assert features["prev_route_progression"] == 1.0
    assert features["prev_route_progress_norm"] == pytest.approx(0.5)
    assert features["route_waypoint_failed_recently"] == 0.0


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
        self.load_snapshot_requests = []

    async def reset_episode(self, *, skill, episode, map, seed, run_id, start=None):
        self.reset_requests.append(
            {"skill": skill, "episode": episode, "map": map, "seed": seed, "start": start}
        )
        return EpisodeReset(
            accepted=True,
            message="queued",
            skill=skill,
            episode=episode,
            map=map,
            seed=seed,
            seed_applied=False,
            start_queued=False,
        )

    async def load_snapshot(self, *, slot, run_id=""):
        self.load_snapshot_requests.append({"slot": slot, "run_id": run_id})
        return SimpleNamespace(
            accepted=True,
            message="queued",
            slot=slot,
            save_queued=False,
            load_queued=True,
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


def _state(
    *,
    tick=1,
    level_time=None,
    kills=0,
    enemy=False,
    enemy_line_of_sight=True,
    combat=False,
    enemy_distance=256,
    health=100,
    hazard=False,
    route=False,
    contact_use=False,
    contact_use_distance_units=640,
    direction_probes=None,
    x_units=0,
    y_units=0,
):
    position = SimpleNamespace(x_fp=x_units * 65536, y_fp=y_units * 65536, z_fp=0)
    obj = SimpleNamespace(
        id=1,
        position=position,
        angle_degrees=0,
        height_fp=0,
        health=health,
        type_id=0,
        internal_type=0,
        flags=0,
        attacking_id=0,
        distance_fp=0,
    )
    player = SimpleNamespace(
        object=obj,
        health=health,
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
        enemies.append(
            SimpleNamespace(object=enemy_obj, line_of_sight=enemy_line_of_sight, target_id=0)
        )
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
        current_sector=SimpleNamespace(
            sector_id=4 if hazard else 0,
            special=5 if hazard else 0,
            floor_height_fp=-24 * 65536 if hazard else 0,
            ceiling_height_fp=128 * 65536 if hazard else 0,
            light_level=160 if hazard else 0,
            damaging=hazard,
            damage_per_32_tics=10 if hazard else 0,
            exit_damage=False,
        ),
        route_waypoint=None,
    )
    if direction_probes is not None:
        navigation.direction_probes = [
            SimpleNamespace(
                angle_offset_degrees=probe["angle_offset_degrees"],
                open=probe.get("open", True),
                block_distance_fp=int(probe.get("block_distance", 128) * 65536),
                blocking_line_special=probe.get("blocking_line_special", 0),
                use_line_ahead=probe.get("use_line_ahead", False),
            )
            for probe in direction_probes
        ]
    if contact_use:
        contact_x_fp = int(contact_use_distance_units * 65536)
        navigation.use_lines = [
            SimpleNamespace(
                line_id=151,
                midpoint=SimpleNamespace(x_fp=contact_x_fp, y_fp=0, z_fp=0),
                start=SimpleNamespace(x_fp=contact_x_fp, y_fp=-64 * 65536, z_fp=0),
                end=SimpleNamespace(x_fp=contact_x_fp, y_fp=64 * 65536, z_fp=0),
                nearest_point=SimpleNamespace(x_fp=contact_x_fp, y_fp=0, z_fp=0),
                special=1,
                tag=0,
                distance_fp=contact_x_fp,
                nearest_distance_fp=contact_x_fp,
            )
        ]
    if route:
        route_distance_units = abs(512 - x_units)
        route_line = SimpleNamespace(
            line_id=88,
            midpoint=SimpleNamespace(x_fp=512 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=512 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=512 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=512 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=1,
            distance_fp=route_distance_units * 65536,
            nearest_distance_fp=route_distance_units * 65536,
        )
        navigation.use_lines.append(route_line)
        navigation.route_waypoint = SimpleNamespace(
            line=route_line,
            priority=0,
            exit=False,
            walk_trigger=True,
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
            level_time=tick if level_time is None else level_time,
            total_kills=1,
            total_items=0,
            total_secrets=0,
            gamestate=0,
        ),
        has_delta_state=False,
        navigation=navigation,
        combat=combat_probe,
    )
