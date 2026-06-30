import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from restfuldoom_agent.client import EpisodeReset
from restfuldoom_agent.client import agent_pb2
from restfuldoom_agent.brain import (
    AgentMemory,
    BT_ATTACK,
    BT_USE,
    LINE_ATTEMPT_STALL_TICS,
    _NON_LOCOMOTION_SKILLS,
    extract_features,
)
from restfuldoom_agent.env import (
    DoomAgentEnv,
    DoomEnvConfig,
    EXIT_ROUTE_FAILURE_RECOVERY_THRESHOLD,
    LOW_HEALTH_RETREAT_STREAK_LIMIT,
    RECOVER_STUCK_ROUTE_STREAK_LIMIT,
    SKILL_ACTIONS,
    SkillController,
    VISIBLE_CONTACT_RETREAT_STREAK_LIMIT,
    _route_outcome,
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
    first = _state(tick=1, kills=0, enemy=True, combat=True)
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


def test_doom_agent_env_reset_waits_for_ready_level_time():
    early = _state(tick=1, level_time=1, enemy=True)
    ready = _state(tick=5, level_time=5, enemy=True)
    client = _StreamingFakeClient([early, ready])
    env = DoomAgentEnv(
        DoomEnvConfig(goal_preset="combat", reset_ready_level_time=5),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=99)
        await env.close()
        return env._current_state

    state = asyncio.run(run())

    assert state.level.level_time == 5
    assert state.tick == 5


def test_doom_agent_env_terminates_on_required_kills_after_reset():
    first = _state(tick=1, kills=0, enemy=True, combat=True)
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


def test_doom_agent_env_step_enforces_action_mask_for_disallowed_skill():
    first = _state(tick=1, enemy=True, combat=False)
    second = _state(tick=2, enemy=True, combat=False, enemy_distance=220)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(max_steps=10, goal_preset="combat", max_action_tics=1),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=99)
        step = await env.step(SKILL_ACTIONS.index("fire"))
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["requested_skill"] == "fire"
    assert step.info["skill"] == "close_visible_contact"
    assert step.info["action_mask_enforced"]
    assert not step.info["action_mask_requested_allowed"]
    assert step.info["action_mask_fallback_applied"]
    assert step.info["action_mask_fallback_reason"] == "requested_action_masked"
    assert step.info["action_mask_fallback_skill"] == "close_visible_contact"
    assert step.info["action_index"] == SKILL_ACTIONS.index("close_visible_contact")
    assert step.info["requested_action_index"] == SKILL_ACTIONS.index("fire")


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
    assert not visible_mask["seek_enemy"]
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


def test_skill_controller_healthy_close_visible_contact_caps_retreat_loop():
    controller = SkillController()
    state = _state(enemy=True, combat=False, health=100, enemy_distance=200)

    initial_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    controller._previous_action_index = SKILL_ACTIONS.index("retreat")
    controller._same_skill_streak = VISIBLE_CONTACT_RETREAT_STREAK_LIMIT
    capped_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert initial_mask["close_visible_contact"]
    assert initial_mask["retreat"]
    assert capped_mask["close_visible_contact"]
    assert not capped_mask["retreat"]


def test_skill_controller_low_health_visible_contact_respects_retreat_loop_cap():
    controller = SkillController()
    state = _state(enemy=True, combat=False, health=30, enemy_distance=200)

    controller._previous_action_index = SKILL_ACTIONS.index("retreat")
    controller._same_skill_streak = LOW_HEALTH_RETREAT_STREAK_LIMIT
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["close_visible_contact"]
    assert not mask["retreat"]


def test_skill_controller_stuck_low_health_visible_contact_blocks_retreat():
    controller = SkillController()
    state = _state(
        tick=controller.params.stuck_window_tics + 5,
        enemy=True,
        combat=False,
        health=30,
        enemy_distance=200,
    )
    controller.policy._last_position = (0.0, 0.0)
    controller.policy._last_progress_tick = 0

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["close_visible_contact"]
    assert mask["recover_stuck"]
    assert not mask["retreat"]


def test_skill_controller_low_health_damaging_sector_combat_allows_fire():
    controller = SkillController()
    state = _state(enemy=True, combat=True, health=5, hazard=True)

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["retreat"]


def test_skill_controller_critical_shootable_contact_allows_fire_off_hazard():
    controller = SkillController()
    state = _state(enemy=True, combat=True, health=15, hazard=False)

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["retreat"]
    assert not mask["route_progression"]
    assert not mask["press_exit"]


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


def test_skill_controller_low_health_stale_contact_keeps_recovery_window():
    controller = SkillController()
    first = _state(tick=5, enemy=True, combat=False)
    low_health_lost_contact = _state(
        tick=40,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        health=31,
        kills=4,
        route=True,
    )

    controller.action_mask(first)
    controller._previous_action_index = SKILL_ACTIONS.index("retreat")
    controller._same_skill_streak = LOW_HEALTH_RETREAT_STREAK_LIMIT
    break_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(low_health_lost_contact)))
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 8
    recovery_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(low_health_lost_contact)))

    assert break_mask["close_visible_contact"]
    assert not break_mask["retreat"]
    assert recovery_mask["close_visible_contact"]
    assert not recovery_mask["retreat"]


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


def test_skill_controller_allows_late_contact_recovery_for_known_enemy(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [512.0, 0.0],
            "last_distance": 512.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(tick=40, kills=3, enemy=True, combat=False)
    lost_contact = _state(
        tick=60,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
    )

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["close_visible_contact"]
    assert not mask["seek_enemy"]


def test_skill_controller_breaks_late_contact_loop_with_route_recovery(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1800.0, 0.0],
            "last_distance": 1800.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(tick=40, kills=3, enemy=True, combat=False)
    lost_contact = _state(
        tick=120,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1800,
        combat=False,
        route=True,
    )

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 24
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["route_progression"]
    assert mask["recover_stuck"]
    assert not mask["close_visible_contact"]
    assert not mask["seek_enemy"]


def test_skill_controller_breaks_late_contact_loop_near_route_after_stale_close(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1190.0, 0.0],
            "last_distance": 1190.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(tick=40, kills=3, enemy=True, combat=False)
    lost_contact = _state(
        tick=120,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
    )
    route_line = lost_contact.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 503 * 65536
    route_line.nearest_distance_fp = 503 * 65536

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 23
    close_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert close_mask["close_visible_contact"]

    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 24
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["route_progression"]
    assert mask["recover_stuck"]
    assert not mask["close_visible_contact"]
    assert not mask["seek_enemy"]

    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 1
    sticky_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert sticky_mask["route_progression"]
    assert sticky_mask["recover_stuck"]
    assert not sticky_mask["close_visible_contact"]

    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 8
    recover_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert recover_mask["recover_stuck"]
    assert not recover_mask["route_progression"]
    assert not recover_mask["close_visible_contact"]


def test_skill_controller_late_route_recovery_defers_to_contact_use_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1190.0, 0.0],
            "last_distance": 1190.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(
        tick=40,
        kills=3,
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=900,
    )
    contact_line = visible_contact.navigation.use_lines[0]
    controller._remember_contact_use_line(
        extract_features(visible_contact, controller.memory, controller.params),
        {
            "line_id": contact_line.line_id,
            "special": contact_line.special,
            "distance": 900.0,
            "angle_delta": 0.0,
        },
    )
    lost_contact = _state(
        tick=120,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
        contact_use=True,
        contact_use_distance_units=900,
    )
    route_line = lost_contact.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 520 * 65536
    route_line.nearest_distance_fp = 520 * 65536

    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 1
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["close_visible_contact"]
    assert not mask["route_progression"]
    assert not mask["recover_stuck"]


def test_skill_controller_stale_late_contact_use_line_yields_to_route_recovery(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1190.0, 0.0],
            "last_distance": 1190.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(
        tick=40,
        kills=3,
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=900,
    )
    contact_line = visible_contact.navigation.use_lines[0]
    controller._remember_contact_use_line(
        extract_features(visible_contact, controller.memory, controller.params),
        {
            "line_id": contact_line.line_id,
            "special": contact_line.special,
            "distance": 900.0,
            "angle_delta": 0.0,
        },
    )
    lost_contact = _state(
        tick=120,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
        contact_use=True,
        contact_use_distance_units=900,
    )
    route_line = lost_contact.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 520 * 65536
    route_line.nearest_distance_fp = 520 * 65536

    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 63
    contact_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert contact_mask["close_visible_contact"]
    assert not contact_mask["route_progression"]
    assert not contact_mask["recover_stuck"]

    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 64
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["route_progression"]
    assert not mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert not mask["seek_enemy"]

    controller._previous_action_index = SKILL_ACTIONS.index("recover_stuck")
    controller._same_skill_streak = 16
    latched_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert latched_mask["route_progression"]
    assert not latched_mask["close_visible_contact"]


def test_skill_controller_breaks_expired_late_contact_loop_after_long_close(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1190.0, 0.0],
            "last_distance": 1190.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(tick=40, kills=3, enemy=True, combat=False)
    lost_contact = _state(
        tick=700,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
    )
    route_line = lost_contact.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 790 * 65536
    route_line.nearest_distance_fp = 790 * 65536

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 63
    close_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert close_mask["close_visible_contact"]

    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 64
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["route_progression"]
    assert mask["recover_stuck"]
    assert not mask["close_visible_contact"]


def test_skill_controller_route_progression_uses_late_contact_route_waypoint():
    controller = SkillController()
    controller.policy._start_kills = 0
    controller._failed_route_attempt_count = EXIT_ROUTE_FAILURE_RECOVERY_THRESHOLD
    state = _state(
        tick=120,
        kills=4,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
        contact_use=True,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.tag = 2
    route_line.midpoint.x_fp = -800 * 65536
    route_line.midpoint.y_fp = 160 * 65536
    route_line.start.x_fp = -800 * 65536
    route_line.start.y_fp = 96 * 65536
    route_line.end.x_fp = -800 * 65536
    route_line.end.y_fp = 224 * 65536
    route_line.nearest_point.x_fp = -800 * 65536
    route_line.nearest_point.y_fp = 160 * 65536
    route_line.distance_fp = 816 * 65536
    route_line.nearest_distance_fp = 816 * 65536

    _action, decision = controller.action_for(
        SKILL_ACTIONS.index("route_progression"),
        state,
    )

    assert decision["ppo_skill"] == "route_progression"
    assert decision["skill"] != "break_cell_loop"
    assert decision["skill"] != "unstick_forward"
    assert decision["use_line"]["line_id"] == 195


def test_skill_controller_damaging_route_waypoint_ignores_stale_blacklist():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=120,
        kills=4,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1190,
        combat=False,
        route=True,
        hazard=True,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.tag = 2
    route_line.midpoint.x_fp = 128 * 65536
    route_line.midpoint.y_fp = 790 * 65536
    route_line.start.x_fp = 64 * 65536
    route_line.start.y_fp = 790 * 65536
    route_line.end.x_fp = 192 * 65536
    route_line.end.y_fp = 790 * 65536
    route_line.nearest_point.x_fp = 128 * 65536
    route_line.nearest_point.y_fp = 790 * 65536
    route_line.distance_fp = 800 * 65536
    route_line.nearest_distance_fp = 800 * 65536
    features = extract_features(state, controller.memory, controller.params)
    line = features.navigation["route_waypoint"]["line"]
    controller.policy._blocked_use_lines[
        controller.policy._line_attempt_key(features, line)
    ] = state.tick + 1000

    _action, decision = controller.action_for(
        SKILL_ACTIONS.index("route_progression"),
        state,
    )

    assert decision["ppo_skill"] == "route_progression"
    assert decision["skill"] != "break_cell_loop"
    assert decision["use_line"]["line_id"] == 195


def test_skill_controller_late_contact_use_line_yields_to_route_recovery(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [1800.0, 0.0],
            "last_distance": 1800.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(
        tick=40,
        kills=3,
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=64,
    )
    lost_contact = _state(
        tick=120,
        kills=3,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=1800,
        combat=False,
        route=True,
        contact_use=True,
        contact_use_distance_units=80,
    )
    lost_contact.navigation.route_waypoint.line.line_id = 195
    lost_contact.navigation.route_waypoint.line.special = 88
    lost_contact.navigation.route_waypoint.line.distance_fp = 1550 * 65536
    lost_contact.navigation.route_waypoint.line.nearest_distance_fp = 1550 * 65536

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 12
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert mask["close_visible_contact"]
    assert not mask["route_progression"]

    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 32
    recovery_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert recovery_mask["route_progression"]
    assert recovery_mask["recover_stuck"]
    assert not recovery_mask["close_visible_contact"]


def test_skill_controller_suppresses_known_enemy_seek_after_exit_kills(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [512.0, 0.0],
            "last_distance": 512.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    visible_contact = _state(tick=40, kills=5, enemy=True, combat=False)
    lost_contact = _state(
        tick=60,
        kills=5,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
    )

    controller.action_for(SKILL_ACTIONS.index("close_visible_contact"), visible_contact)
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(lost_contact)))

    assert not mask["seek_enemy"]


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


def test_doom_agent_env_strict_allowed_skill_filter_prefers_allowed_heuristic():
    class HeuristicFireController:
        def action_mask(self, state):
            return [skill == "route_progression" for skill in SKILL_ACTIONS]

        def heuristic_action_index(self, state):
            return SKILL_ACTIONS.index("fire")

    env = DoomAgentEnv(
        DoomEnvConfig(
            allowed_skills=("close_visible_contact", "fire"),
            strict_allowed_skills=True,
        ),
        controller=HeuristicFireController(),
    )

    env._current_state = _state(enemy=True, combat=True)
    mask = dict(zip(SKILL_ACTIONS, env.action_mask()))

    assert mask["fire"]
    assert not mask["close_visible_contact"]
    assert not mask["route_progression"]
    assert env._last_action_mask_filter["fallback_applied"]
    assert env._last_action_mask_filter["fallback_skill"] == "fire"
    assert env._last_action_mask_filter["fallback_reason"] == "heuristic_allowed_skill"


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
    assert not mask["seek_enemy"]
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


def test_skill_controller_far_stale_contact_route_failure_restores_progression():
    controller = SkillController()
    controller.policy._start_kills = 0
    first = _state(tick=5, kills=1, enemy=True, combat=False)
    route_state = _state(
        tick=20,
        kills=1,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        enemy_distance=1000,
        route=True,
    )
    enemy_obj = route_state.enemies[0].object
    enemy_obj.position.x_fp = 1000 * 65536
    enemy_obj.distance_fp = 1000 * 65536
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
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(route_state)))

    assert mask["close_visible_contact"]
    assert mask["route_progression"]


def test_skill_controller_recent_contact_does_not_suppress_exit_route():
    controller = SkillController()
    controller.policy._start_kills = 0
    first = _state(tick=5, kills=1, enemy=True, combat=False)
    active = _state(
        tick=20,
        kills=5,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        route=True,
        route_exit=True,
        x_units=-1400,
    )
    line = active.navigation.route_waypoint.line
    line.line_id = 330
    line.special = 11
    active.navigation.route_waypoint.exit = True
    active.navigation.route_waypoint.walk_trigger = False
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
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(active)))

    assert mask["close_visible_contact"]
    assert mask["route_progression"]
    assert mask["press_exit"]


def test_skill_controller_low_health_stale_contact_preserves_exit_route():
    controller = SkillController()
    controller.policy._start_kills = 0
    first = _state(tick=5, kills=4, enemy=True, combat=False, health=31)
    active = _state(
        tick=20,
        kills=5,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        health=31,
        route=True,
        route_exit=True,
    )
    line = active.navigation.route_waypoint.line
    line.line_id = 330
    line.special = 11
    active.navigation.route_waypoint.exit = True
    active.navigation.route_waypoint.walk_trigger = False

    controller.action_mask(first)
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(active)))

    assert mask["route_progression"]
    assert not mask["retreat"]


def test_skill_controller_postcombat_blocked_exit_line_exposes_press_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=40,
        kills=5,
        combat=False,
        health=72,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=528 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=528 * 65536, y_fp=-32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=528 * 65536, y_fp=32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=528 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=528 * 65536,
            nearest_distance_fp=528 * 65536,
        )
    ]

    features = extract_features(state, controller.memory, controller.params)
    exit_line = features.navigation["use_lines"][0]
    for key in controller.policy._line_block_keys(features, exit_line):
        controller.policy._blocked_use_lines[key] = features.tick + 1000

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["press_exit"]


def test_skill_controller_postcombat_damaging_stale_contact_exposes_far_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    visible_contact = _state(
        tick=40,
        kills=4,
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=300,
    )
    controller.action_mask(visible_contact)
    state = _state(
        tick=80,
        kills=5,
        health=36,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
        hazard=True,
        contact_use=True,
        contact_use_distance_units=624,
    )
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1400 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1400 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1400 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1400 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1400 * 65536,
            nearest_distance_fp=1400 * 65536,
        )
    )
    features = extract_features(state, controller.memory, controller.params)
    exit_line = next(
        line for line in features.navigation["use_lines"] if line["line_id"] == 330
    )
    for key in controller.policy._line_block_keys(features, exit_line):
        controller.policy._blocked_use_lines[key] = features.tick + 1000

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["press_exit"]
    assert mask["close_visible_contact"]


def test_skill_controller_press_exit_uses_blocked_postcombat_exit_line():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=40, kills=5, combat=False, health=72)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=528 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=528 * 65536, y_fp=-32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=528 * 65536, y_fp=32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=528 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=528 * 65536,
            nearest_distance_fp=528 * 65536,
        )
    ]

    features = extract_features(state, controller.memory, controller.params)
    exit_line = features.navigation["use_lines"][0]
    for key in controller.policy._line_block_keys(features, exit_line):
        controller.policy._blocked_use_lines[key] = features.tick + 1000

    _action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert decision["use_line"]["line_id"] == 330


def test_skill_controller_failed_stale_exit_route_exposes_recovery():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=80, kills=5, route=True, route_exit=False, x_units=-1400)
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2912 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=2912 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2912 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2912 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2200 * 65536,
            nearest_distance_fp=2200 * 65536,
        )
    )
    route_index = SKILL_ACTIONS.index("route_progression")

    assert dict(zip(SKILL_ACTIONS, controller.action_mask(state)))["route_progression"]

    for _ in range(4):
        controller.record_action_history(
            action_index=route_index,
            had_shootable_target=False,
            route_outcome={
                "attempted": True,
                "line_id": 330,
                "exit": True,
                "walk_trigger": False,
                "progress_units": -1.0,
                "reached": False,
                "failed": True,
            },
        )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["recover_stuck"]
    assert mask["route_progression"]


def test_skill_controller_route_progression_recovers_after_failed_stale_exit_route():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=80, kills=5, route=True, route_exit=False, x_units=-1400)
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    state.navigation.forward_open = True
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2912 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=2912 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2912 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2912 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2200 * 65536,
            nearest_distance_fp=2200 * 65536,
        )
    )
    route_index = SKILL_ACTIONS.index("route_progression")
    for _ in range(4):
        controller.record_action_history(
            action_index=route_index,
            had_shootable_target=False,
            route_outcome={
                "attempted": True,
                "line_id": 330,
                "exit": True,
                "walk_trigger": False,
                "progress_units": -1.0,
                "reached": False,
                "failed": True,
            },
        )

    action, decision = controller.action_for(route_index, state)

    assert decision["ppo_skill"] == "route_progression"
    assert decision["skill"] in {
        "unstick_approach_exit_line",
        "unstick_route_to_exit_line",
    }
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_failed_far_walk_route_yields_to_local_use_recovery():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=80,
        kills=2,
        route=True,
        contact_use=True,
        contact_use_distance_units=64,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 1500 * 65536
    route_line.nearest_distance_fp = 1500 * 65536
    route_index = SKILL_ACTIONS.index("route_progression")

    assert dict(zip(SKILL_ACTIONS, controller.action_mask(state)))["route_progression"]

    for _ in range(4):
        controller.record_action_history(
            action_index=route_index,
            had_shootable_target=False,
            route_outcome={
                "attempted": True,
                "line_id": 195,
                "exit": False,
                "walk_trigger": True,
                "progress_units": -1.0,
                "reached": False,
                "failed": True,
            },
        )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(route_index, state)

    assert mask["open_use_line"]
    assert mask["recover_stuck"]
    assert not mask["route_progression"]
    assert decision["ppo_skill"] == "route_progression"
    assert decision["skill"] == "unstick_use"
    assert action.action == agent_pb2.ACTION_USE


def test_skill_controller_masks_far_exit_as_route_not_press():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=1, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=800 * 65536, y_fp=-32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=800 * 65536, y_fp=32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=800 * 65536,
            nearest_distance_fp=800 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["route_progression"]
    assert not mask["press_exit"]
    assert controller.heuristic_action_index(state) == SKILL_ACTIONS.index("route_progression")


def test_skill_controller_keeps_midrange_exit_out_of_generic_open_use():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=1, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=480 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=480 * 65536, y_fp=-32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=480 * 65536, y_fp=32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=480 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=480 * 65536,
            nearest_distance_fp=480 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["route_progression"]
    assert not mask["open_use_line"]
    assert not mask["press_exit"]
    assert controller.heuristic_action_index(state) == SKILL_ACTIONS.index("route_progression")


def test_skill_controller_allows_post_combat_press_exit_approach_before_push_window():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=5, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=800 * 65536, y_fp=-32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=800 * 65536, y_fp=32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=800 * 65536,
            nearest_distance_fp=800 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["route_progression"]
    assert mask["press_exit"]
    assert controller.heuristic_action_index(state) == SKILL_ACTIONS.index("press_exit")
    assert decision["use_line"]["line_id"] == 330


def test_skill_controller_allows_post_combat_press_exit_for_far_visible_final_line():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=5, enemy=False, route=True)
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2200 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=2200 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2200 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2200 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2200 * 65536,
            nearest_distance_fp=2200 * 65536,
        )
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["route_progression"]
    assert mask["press_exit"]
    assert controller.heuristic_action_index(state) == SKILL_ACTIONS.index("press_exit")
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_far_final_line_approach_keeps_full_forward_speed():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=5, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1000 * 65536, y_fp=-2500 * 65536, z_fp=0),
            start=SimpleNamespace(x_fp=1000 * 65536, y_fp=-2564 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1000 * 65536, y_fp=-2436 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1000 * 65536, y_fp=-2500 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2692 * 65536,
            nearest_distance_fp=2692 * 65536,
        )
    ]

    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert decision["skill"] == "approach_far_exit_line"
    assert action.raw.forward_move == controller.params.move_amount


def test_skill_controller_critical_combat_probe_forces_fire_before_post_combat_press_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=14,
        enemy=True,
        enemy_line_of_sight=False,
        combat=True,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=628 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=628 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=628 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=628 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=628 * 65536,
            nearest_distance_fp=628 * 65536,
        )
    ]

    controller.action_mask(_state(tick=1, enemy=True, combat=False))
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["retreat"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]


def test_skill_controller_visible_contact_keeps_near_final_line_legal():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=5, health=43, enemy=True, combat=True)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=792 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=792 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=792 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=792 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=792 * 65536,
            nearest_distance_fp=792 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["fire"]
    assert mask["press_exit"]
    assert not mask["route_progression"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_low_health_visible_contact_defers_far_final_line():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=34,
        enemy=True,
        enemy_line_of_sight=True,
        combat=False,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1880 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1880 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1880 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1880 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1880 * 65536,
            nearest_distance_fp=1880 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["retreat"]
    assert not mask["close_visible_contact"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]


def test_skill_controller_visible_contact_defers_far_final_line_after_kills():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=75,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=1800,
        combat=False,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1812 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1812 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1812 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1812 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1812 * 65536,
            nearest_distance_fp=1812 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["close_visible_contact"]
    assert not mask["retreat"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]


def test_skill_controller_late_visible_contact_blocks_preexit_route():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=4,
        health=75,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=900,
        combat=False,
        route=True,
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["close_visible_contact"]
    assert not mask["route_progression"]
    assert not mask["press_exit"]


def test_skill_controller_low_health_late_visible_contact_keeps_close_option():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=4,
        health=19,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=480,
        combat=False,
        route=True,
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["close_visible_contact"]
    assert mask["retreat"]
    assert not mask["route_progression"]
    assert not mask["press_exit"]


def test_skill_controller_combat_probe_target_allows_fire_without_visible_enemy():
    controller = SkillController()
    state = _state(kills=3, enemy=False, combat=True, route=True)

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("fire"), state)

    assert mask["fire"]
    assert not mask["route_progression"]
    assert decision["skill"] == "ppo_fire"
    assert action.raw.buttons & BT_ATTACK


def test_skill_controller_postcombat_visible_contact_loop_yields_to_route():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=75,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=1800,
        combat=False,
        route=True,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 1812 * 65536
    route_line.nearest_distance_fp = 1812 * 65536
    controller._previous_action_index = SKILL_ACTIONS.index("close_visible_contact")
    controller._same_skill_streak = 24

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["route_progression"]
    assert mask["recover_stuck"]
    assert not mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert not mask["retreat"]

    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 1
    sticky_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert sticky_mask["route_progression"]
    assert sticky_mask["recover_stuck"]
    assert not sticky_mask["close_visible_contact"]

    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 8
    recover_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert recover_mask["recover_stuck"]
    assert not recover_mask["route_progression"]
    assert not recover_mask["close_visible_contact"]


def test_skill_controller_critical_visible_combat_forces_fire_before_far_final_line():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=3,
        enemy=True,
        enemy_line_of_sight=True,
        combat=True,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2280 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=2280 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2280 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2280 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2280 * 65536,
            nearest_distance_fp=2280 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]
    assert not mask["retreat"]


def test_skill_controller_critical_visible_final_line_suppresses_contact_work():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=6,
        enemy=True,
        enemy_line_of_sight=True,
        combat=False,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=228 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=228 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=228 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=228 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=228 * 65536,
            nearest_distance_fp=228 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["press_exit"]
    assert not mask["route_progression"]
    assert not mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert not mask["retreat"]


def test_skill_controller_postcombat_exit_commitment_suppresses_stale_contact_work(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["enemies"] = {
        "7": {
            "last_seen_tick": 40,
            "last_position": [900.0, 0.0],
            "last_distance": 900.0,
            "last_health": 20,
            "line_of_sight": True,
        }
    }
    controller = SkillController(memory=memory)
    controller.policy._start_kills = 0
    visible_contact = _state(
        tick=40,
        kills=4,
        enemy=True,
        combat=False,
        contact_use=True,
        contact_use_distance_units=300,
    )
    controller.action_mask(visible_contact)

    state = _state(
        tick=120,
        kills=5,
        health=38,
        enemy=True,
        enemy_line_of_sight=False,
        enemy_distance=900,
        combat=False,
        contact_use=True,
        contact_use_distance_units=300,
    )
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=176 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=176 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=176 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=176 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=176 * 65536,
            nearest_distance_fp=176 * 65536,
        )
    )
    controller._previous_action_index = SKILL_ACTIONS.index("open_use_line")
    controller._same_skill_streak = 1

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["press_exit"]
    assert mask["route_progression"]
    assert not mask["close_visible_contact"]
    assert not mask["open_use_line"]
    assert not mask["seek_enemy"]
    assert not mask["retreat"]


def test_skill_controller_critical_walk_route_suppresses_far_press_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=120,
        kills=5,
        health=7,
        combat=False,
        route=True,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 300 * 65536
    route_line.nearest_distance_fp = 300 * 65536
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1865 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1865 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1865 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1865 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1865 * 65536,
            nearest_distance_fp=1865 * 65536,
        )
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["route_progression"]
    assert not mask["press_exit"]


def test_skill_controller_critical_far_walk_route_keeps_far_press_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=120,
        kills=5,
        health=13,
        combat=False,
        hazard=True,
        route=True,
    )
    route_line = state.navigation.route_waypoint.line
    route_line.line_id = 195
    route_line.special = 88
    route_line.distance_fp = 528 * 65536
    route_line.nearest_distance_fp = 528 * 65536
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1865 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1865 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1865 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1865 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1865 * 65536,
            nearest_distance_fp=1865 * 65536,
        )
    )

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["route_progression"]
    assert mask["press_exit"]


def test_skill_controller_critical_combat_probe_forces_fire_before_far_post_combat_press_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=14,
        enemy=True,
        enemy_line_of_sight=False,
        combat=True,
    )
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2200 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=2200 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2200 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2200 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2200 * 65536,
            nearest_distance_fp=2200 * 65536,
        )
    ]

    controller.action_mask(_state(tick=1, enemy=True, combat=False))
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["retreat"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]


def test_skill_controller_critical_hazard_keeps_blocked_final_line_legal():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=40,
        kills=5,
        health=18,
        hazard=True,
        enemy=True,
        enemy_line_of_sight=False,
        combat=True,
        route=True,
    )
    state.navigation.route_waypoint.line.line_id = 195
    state.navigation.route_waypoint.line.special = 88
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1378 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1378 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1378 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1378 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1378 * 65536,
            nearest_distance_fp=1378 * 65536,
        )
    )
    controller.policy._is_line_blocked = (
        lambda _features, line: int(line.get("line_id", 0)) == 330
    )

    controller.action_mask(_state(tick=1, enemy=True, combat=False))
    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["fire"]
    assert mask["route_progression"]
    assert mask["press_exit"]
    assert not mask["retreat"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_critical_exit_commitment_suppresses_fire_for_final_escape():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=7,
        enemy=True,
        enemy_line_of_sight=True,
        combat=True,
        route=True,
    )
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=668 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=668 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=668 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=668 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=668 * 65536,
            nearest_distance_fp=668 * 65536,
        )
    )
    controller._previous_action_index = SKILL_ACTIONS.index("press_exit")
    controller._same_skill_streak = 1

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert not mask["fire"]
    assert mask["press_exit"]
    assert mask["route_progression"]
    assert not mask["retreat"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_critical_exit_commitment_survives_intervening_fire():
    controller = SkillController()
    controller.policy._start_kills = 0
    committed = _state(
        tick=100,
        kills=5,
        health=14,
        enemy=False,
    )
    committed.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=760 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=760 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=760 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=760 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=760 * 65536,
            nearest_distance_fp=760 * 65536,
        )
    ]
    controller.action_for(SKILL_ACTIONS.index("press_exit"), committed)

    threatened = _state(
        tick=140,
        kills=5,
        health=14,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=420,
        combat=True,
    )
    threatened.navigation.use_lines = committed.navigation.use_lines
    threatened.combat.target_distance_fp = 420 * 65536
    controller._previous_action_index = SKILL_ACTIONS.index("fire")
    controller._same_skill_streak = 12

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(threatened)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), threatened)

    assert not mask["fire"]
    assert mask["press_exit"]
    assert mask["route_progression"]
    assert not mask["retreat"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_critical_visible_exit_focus_overrides_retreat():
    controller = SkillController()
    controller.policy._start_kills = 0
    committed = _state(
        tick=100,
        kills=5,
        health=18,
        enemy=False,
    )
    committed.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1112 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1112 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1112 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1112 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1112 * 65536,
            nearest_distance_fp=1112 * 65536,
        )
    ]
    controller.action_for(SKILL_ACTIONS.index("press_exit"), committed)

    threatened = _state(
        tick=140,
        kills=5,
        health=13,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=64,
        combat=False,
    )
    threatened.navigation.use_lines = committed.navigation.use_lines

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(threatened)))
    action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), threatened)

    assert mask["press_exit"]
    assert not mask["retreat"]
    assert not mask["close_visible_contact"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_low_health_stale_contact_keeps_committed_far_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    visible_contact = _state(
        tick=80,
        kills=5,
        health=12,
        enemy=True,
        enemy_line_of_sight=True,
        combat=False,
    )
    controller.action_mask(visible_contact)
    committed = _state(tick=100, kills=5, health=12, enemy=False)
    committed.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1900 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=1900 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=1900 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1900 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1900 * 65536,
            nearest_distance_fp=1900 * 65536,
        )
    ]
    controller.action_for(SKILL_ACTIONS.index("press_exit"), committed)
    controller.record_action_history(
        action_index=SKILL_ACTIONS.index("press_exit"),
        had_shootable_target=False,
    )
    stale_contact = _state(
        tick=140,
        kills=5,
        health=12,
        enemy=True,
        enemy_line_of_sight=False,
        combat=False,
    )
    stale_contact.navigation.use_lines = committed.navigation.use_lines

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(stale_contact)))
    action, decision = controller.action_for(
        SKILL_ACTIONS.index("press_exit"),
        stale_contact,
    )

    assert mask["press_exit"]
    assert not mask["retreat"]
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0


def test_skill_controller_mid_critical_shootable_blocks_distant_final_escape():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        kills=5,
        health=17,
        enemy=True,
        enemy_line_of_sight=True,
        combat=True,
        route=True,
    )
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=616 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=616 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=616 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=616 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=616 * 65536,
            nearest_distance_fp=616 * 65536,
        )
    )
    controller._previous_action_index = SKILL_ACTIONS.index("route_progression")
    controller._same_skill_streak = 12

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["fire"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]
    assert not mask["retreat"]


def test_skill_controller_low_health_close_fire_uses_defensive_movement():
    controller = SkillController()
    state = _state(
        health=17,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=164,
        combat=True,
    )
    state.combat.target_distance_fp = 164 * 65536

    action, decision = controller.action_for(SKILL_ACTIONS.index("fire"), state)

    assert decision["skill"] == "ppo_defensive_fire"
    assert action.raw.buttons & BT_ATTACK
    assert action.raw.forward_move < 0
    assert action.raw.side_move != 0


def test_skill_controller_critical_health_far_fire_uses_defensive_movement():
    controller = SkillController()
    state = _state(
        health=9,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=536,
        combat=True,
    )
    state.combat.target_distance_fp = 536 * 65536

    action, decision = controller.action_for(SKILL_ACTIONS.index("fire"), state)

    assert decision["skill"] == "ppo_defensive_fire"
    assert action.raw.buttons & BT_ATTACK
    assert action.raw.forward_move < 0
    assert action.raw.side_move != 0
    assert action.duration_tics == 2


def test_skill_controller_low_health_shootable_cooldown_defers_distant_exit():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(
        tick=100,
        kills=5,
        health=17,
        enemy=True,
        enemy_line_of_sight=True,
        enemy_distance=120,
        combat=True,
        route=True,
    )
    state.combat.target_distance_fp = 120 * 65536
    state.navigation.use_lines.append(
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=576 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=576 * 65536, y_fp=-64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=576 * 65536, y_fp=64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=576 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=576 * 65536,
            nearest_distance_fp=576 * 65536,
        )
    )
    controller.policy._last_shot_tick = state.tick
    controller._previous_action_index = SKILL_ACTIONS.index("fire")
    controller._same_skill_streak = 15

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["retreat"]
    assert not mask["fire"]
    assert not mask["press_exit"]
    assert not mask["route_progression"]


def test_skill_controller_allows_press_exit_inside_activation_range():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=1, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=80 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=80 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=80 * 65536, y_fp=-32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=80 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=80 * 65536,
            nearest_distance_fp=80 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))

    assert mask["press_exit"]
    assert controller.heuristic_action_index(state) == SKILL_ACTIONS.index("press_exit")


def test_skill_controller_approaches_exit_before_push_window():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(kills=1, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=250 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=250 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=250 * 65536, y_fp=-32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=250 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=250 * 65536,
            nearest_distance_fp=250 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["press_exit"]
    assert decision["skill"] == "approach_exit_switch_front"


def test_skill_controller_approaches_postcombat_exit_at_front_plateau():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=20, kills=5, enemy=False)
    state.navigation.forward_open = False
    features = extract_features(state, controller.memory, controller.params)
    line = {
        "line_id": 330,
        "special": 11,
        "tag": 0,
        "distance": 176.0,
        "angle_delta": -24.0,
        "side": 0,
        "front_distance": 148.0,
        "front_angle_delta": 12.0,
    }

    action, decision = controller.policy._advance_progression_line(
        features,
        line,
        stuck=False,
    )

    assert decision["skill"] == "approach_exit_switch_front"
    assert not action.raw.buttons
    assert action.raw.forward_move > 0
    assert action.duration_tics == 4


def test_skill_controller_pushes_stalled_postcombat_exit_front_plateau():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=80, kills=5, enemy=False)
    state.navigation.forward_open = False
    features = extract_features(state, controller.memory, controller.params)
    line = {
        "line_id": 330,
        "special": 11,
        "tag": 0,
        "distance": 176.0,
        "angle_delta": -24.0,
        "side": 0,
        "front_distance": 148.0,
        "front_angle_delta": 12.0,
    }
    controller.policy._exit_push_attempts[
        controller.policy._line_key(features.cell, line)
    ] = {
        "first_tick": features.tick - LINE_ATTEMPT_STALL_TICS - 1,
        "best_distance": 176.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }

    action, decision = controller.policy._advance_progression_line(
        features,
        line,
        stuck=False,
    )

    assert decision["skill"] == "push_exit_switch"
    assert action.raw.buttons & BT_USE
    assert action.raw.forward_move > 0


def test_skill_controller_critical_aligned_exit_skips_local_assist_door():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=120, kills=6, health=2, enemy=False)
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=464 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=464 * 65536, y_fp=64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=464 * 65536, y_fp=-64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=464 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=464 * 65536,
            nearest_distance_fp=464 * 65536,
        ),
        SimpleNamespace(
            line_id=324,
            midpoint=SimpleNamespace(x_fp=370 * 65536, y_fp=96 * 65536, z_fp=0),
            start=SimpleNamespace(x_fp=370 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=370 * 65536, y_fp=160 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=370 * 65536, y_fp=96 * 65536),
            special=1,
            tag=0,
            distance_fp=382 * 65536,
            nearest_distance_fp=382 * 65536,
        ),
    ]
    features = extract_features(state, controller.memory, controller.params)
    exit_line = dict(
        next(
            line
            for line in features.navigation["use_lines"]
            if int(line["line_id"]) == 330
        )
    )
    exit_line["side"] = 0
    exit_line["angle_delta"] = 0.0
    exit_line["front_angle_delta"] = 8.0
    exit_line["front_distance"] = 450.0

    action, decision = controller.policy._advance_progression_line(
        features,
        exit_line,
        stuck=False,
    )

    assert decision["use_line"]["line_id"] == 330
    assert decision["skill"] == "approach_progression_line"
    assert action.raw.forward_move > 0


def test_skill_controller_direct_aligned_exit_skips_local_door_before_critical_health():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=120, kills=5, health=81, enemy=False)
    state.navigation.forward_open = False
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=227 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=227 * 65536, y_fp=64 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=227 * 65536, y_fp=-64 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=227 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=227 * 65536,
            nearest_distance_fp=227 * 65536,
        ),
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=107 * 65536, y_fp=64 * 65536, z_fp=0),
            start=SimpleNamespace(x_fp=107 * 65536, y_fp=0, z_fp=0),
            end=SimpleNamespace(x_fp=107 * 65536, y_fp=128 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=107 * 65536, y_fp=64 * 65536),
            special=1,
            tag=0,
            distance_fp=107 * 65536,
            nearest_distance_fp=107 * 65536,
        ),
    ]
    features = extract_features(state, controller.memory, controller.params)
    exit_line = dict(
        next(
            line
            for line in features.navigation["use_lines"]
            if int(line["line_id"]) == 330
        )
    )
    exit_line["side"] = 0
    exit_line["angle_delta"] = 2.0
    exit_line["front_angle_delta"] = 28.0
    exit_line["front_distance"] = 209.0

    _action, decision = controller.policy._advance_progression_line(
        features,
        exit_line,
        stuck=False,
    )

    assert decision["use_line"]["line_id"] == 330
    assert decision["skill"] != "use_exit_route_local_door"
    assert decision["skill"] != "approach_nearby_use_line"


def test_skill_controller_stuck_aligned_exit_slides_inside_assist_range():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=120, kills=6, health=12, enemy=False)
    state.navigation.left_open = True
    state.navigation.right_open = False
    features = extract_features(state, controller.memory, controller.params)
    line = {
        "line_id": 330,
        "special": 11,
        "tag": 0,
        "distance": 402.0,
        "angle_delta": 0.5,
        "side": 0,
        "front_distance": 410.0,
        "front_angle_delta": 14.0,
    }

    action, decision = controller.policy._advance_progression_line(
        features,
        line,
        stuck=True,
    )

    assert decision["skill"] == "recover_aligned_exit_slide"
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0
    assert action.raw.side_move < 0


def test_skill_controller_recovers_close_exit_push_from_wrong_front_side():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=80, kills=5, enemy=False)
    state.navigation.back_open = True
    features = extract_features(state, controller.memory, controller.params)
    line = {
        "line_id": 330,
        "special": 11,
        "tag": 0,
        "distance": 18.0,
        "angle_delta": 3.0,
        "side": 0,
        "front_distance": 80.0,
        "front_angle_delta": -144.0,
    }
    controller.policy._exit_push_attempts[
        controller.policy._line_key(features.cell, line)
    ] = {
        "first_tick": features.tick - LINE_ATTEMPT_STALL_TICS - 1,
        "best_distance": 18.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }

    action, decision = controller.policy._advance_progression_line(
        features,
        line,
        stuck=False,
    )

    assert decision["skill"] == "recover_exit_switch_front_side"
    assert action.raw.forward_move < 0
    assert not action.raw.buttons


def test_skill_controller_pushes_exit_when_close_front_path_is_blocked():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=20, kills=1, enemy=False)
    state.navigation.forward_open = False
    state.navigation.use_line_ahead = True
    state.navigation.front_blocking_line_special = 1
    state.navigation.front_block_distance_fp = 16 * 65536
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=168 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=168 * 65536, y_fp=-32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["press_exit"]
    assert decision["skill"] == "push_exit_switch"
    assert _action.raw.buttons & BT_USE
    assert _action.duration_tics == 1

    release_state = _state(tick=21, kills=1, enemy=False)
    release_state.navigation.forward_open = state.navigation.forward_open
    release_state.navigation.use_line_ahead = state.navigation.use_line_ahead
    release_state.navigation.front_blocking_line_special = (
        state.navigation.front_blocking_line_special
    )
    release_state.navigation.front_block_distance_fp = (
        state.navigation.front_block_distance_fp
    )
    release_state.navigation.use_lines = state.navigation.use_lines
    release_action, release_decision = controller.action_for(
        SKILL_ACTIONS.index("press_exit"),
        release_state,
    )

    assert release_decision["skill"] == "release_exit_use"
    assert release_action.raw.buttons == 0
    assert release_action.duration_tics == 1


def test_skill_controller_recovers_after_stalled_exit_push():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=20, kills=1, enemy=False)
    state.navigation.forward_open = False
    state.navigation.use_line_ahead = True
    state.navigation.front_blocking_line_special = 1
    state.navigation.front_block_distance_fp = 16 * 65536
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=168 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=168 * 65536, y_fp=-32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]
    _action, first_decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    stalled = _state(tick=20 + LINE_ATTEMPT_STALL_TICS + 1, kills=1, enemy=False)
    stalled.navigation.forward_open = False
    stalled.navigation.use_line_ahead = True
    stalled.navigation.front_blocking_line_special = 1
    stalled.navigation.front_block_distance_fp = 16 * 65536
    stalled.navigation.use_lines = state.navigation.use_lines
    next_action, next_decision = controller.action_for(
        SKILL_ACTIONS.index("press_exit"),
        stalled,
    )

    assert first_decision["skill"] == "push_exit_switch"
    assert next_decision["skill"] == "use_exit_route_blocker_ahead"
    assert next_action.action == agent_pb2.ACTION_USE


def test_skill_controller_turns_to_exit_when_close_blocked_path_is_misaligned():
    controller = SkillController()
    controller.policy._start_kills = 0
    state = _state(tick=20, kills=1, enemy=False)
    state.navigation.forward_open = False
    state.navigation.use_line_ahead = True
    state.navigation.front_blocking_line_special = 1
    state.navigation.front_block_distance_fp = 16 * 65536
    state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=-132 * 65536, z_fp=0),
            start=SimpleNamespace(x_fp=168 * 65536, y_fp=-100 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=168 * 65536, y_fp=-164 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=-132 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=214 * 65536,
            nearest_distance_fp=214 * 65536,
        )
    ]

    mask = dict(zip(SKILL_ACTIONS, controller.action_mask(state)))
    _action, decision = controller.action_for(SKILL_ACTIONS.index("press_exit"), state)

    assert mask["press_exit"]
    assert decision["skill"] == "turn_to_exit_switch"


def test_exit_switch_decisions_do_not_trigger_stuck_recovery():
    assert "push_exit_switch" in _NON_LOCOMOTION_SKILLS
    assert "press_exit_switch" in _NON_LOCOMOTION_SKILLS
    assert "release_exit_use" in _NON_LOCOMOTION_SKILLS
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


def test_skill_controller_caps_recover_stuck_when_route_available():
    controller = SkillController()
    initial = _state(tick=10, route=True)
    stuck_state = _state(tick=40, route=True)

    controller.action_mask(initial)
    initial_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(stuck_state)))
    controller._previous_action_index = SKILL_ACTIONS.index("recover_stuck")
    controller._same_skill_streak = RECOVER_STUCK_ROUTE_STREAK_LIMIT
    capped_mask = dict(zip(SKILL_ACTIONS, controller.action_mask(stuck_state)))

    assert initial_mask["route_progression"]
    assert initial_mask["recover_stuck"]
    assert capped_mask["route_progression"]
    assert not capped_mask["recover_stuck"]


def test_stuck_walk_trigger_route_yields_to_recovery():
    controller = SkillController()
    state = _state(
        tick=40,
        route=True,
        direction_probes=[
            {"angle_offset_degrees": -90, "open": False},
            {"angle_offset_degrees": -30, "open": False},
            {"angle_offset_degrees": 0, "open": False},
            {"angle_offset_degrees": 30, "open": False},
            {"angle_offset_degrees": 90, "open": False},
        ],
    )
    state.navigation.forward_open = False
    state.navigation.left_open = False
    state.navigation.right_open = False
    state.navigation.back_open = True
    features = extract_features(state, controller.memory, controller.params)
    line = features.navigation["route_waypoint"]["line"]

    _action, decision = controller.policy._advance_progression_line(features, line, stuck=True)

    assert decision["skill"].startswith("unstick_")
    assert decision["skill"] != "cross_progression_line"


def test_exit_progression_line_attempts_do_not_blacklist_route_target():
    controller = SkillController()
    controller.policy._start_kills = 0
    features = extract_features(
        _state(tick=20, kills=1, route=True),
        controller.memory,
        controller.params,
    )
    line = features.navigation["route_waypoint"]["line"]
    line["special"] = 11

    assert controller.policy._record_line_attempt(features, line)
    stalled = replace(features, tick=features.tick + LINE_ATTEMPT_STALL_TICS + 1)

    assert controller.policy._record_line_attempt(stalled, line)
    assert controller.policy._select_progression_line(stalled) is not None


def test_walk_trigger_progression_attempts_blacklist_stalled_route_target():
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

    assert not controller.policy._record_line_attempt(stalled, line)
    assert controller.policy._is_line_blocked(stalled, line)


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


def test_doom_agent_env_episode_reset_context_records_seed_applied():
    first = _state(tick=710, level_time=5, enemy=True)
    client = _FakeClient([first], seed_applied=True)
    env = DoomAgentEnv(
        DoomEnvConfig(goal_preset="combat"),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=123)
        await env.close()
        return env._last_reset_context

    reset_context = asyncio.run(run())

    assert reset_context["source"] == "episode"
    assert reset_context["seed_label"] == 123
    assert reset_context["seed_applied"] is True
    assert reset_context["skipped_reset_episode"] is False


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


def test_doom_agent_env_penalizes_visible_contact_loss_before_shootable():
    first = _state(tick=1, enemy=True, combat=False)
    second = _state(tick=2, enemy=False, combat=False)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="custom",
            max_action_tics=1,
            visible_contact_loss_penalty=0.75,
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
    assert step.info["visible_contact_loss_penalty"] == pytest.approx(-0.75)
    assert step.reward == pytest.approx(-0.75)


def test_doom_agent_env_penalizes_route_progression_before_first_shootable():
    first = _state(tick=1, route=True, x_units=0)
    second = _state(tick=2, route=True, x_units=128)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="custom",
            max_action_tics=1,
            route_progress_reward=0.0,
            pre_shootable_route_penalty=0.4,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(SKILL_ACTIONS.index("route_progression"))
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["route_outcome"]["attempted"]
    assert step.info["pre_shootable_route_penalty"] == pytest.approx(-0.4)
    assert step.info["route_action_reward"] == pytest.approx(-0.4)
    assert step.reward == pytest.approx(-0.4)


def test_doom_agent_env_penalizes_route_progression_before_required_kills():
    first = _state(tick=1, kills=2, route=True, x_units=0)
    second = _state(tick=2, kills=2, route=True, x_units=128)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="custom",
            max_action_tics=1,
            required_kills=5,
            route_progress_reward=0.0,
            pre_required_kill_route_penalty=0.7,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(SKILL_ACTIONS.index("route_progression"))
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["route_outcome"]["attempted"]
    assert step.info["pre_required_kill_route_penalty"] == pytest.approx(-0.7)
    assert step.info["route_action_reward"] == pytest.approx(-0.7)
    assert step.reward == pytest.approx(-0.7)


def test_doom_agent_env_allows_route_progression_after_required_kills():
    first = _state(tick=1, kills=0, route=True, x_units=0)
    second = _state(tick=2, kills=5, route=True, x_units=128)
    client = _DurationAwareFakeClient([first, second])
    env = DoomAgentEnv(
        DoomEnvConfig(
            max_steps=10,
            goal_preset="custom",
            max_action_tics=1,
            required_kills=5,
            route_progress_reward=0.0,
            pre_required_kill_route_penalty=0.7,
        ),
        client=client,
        controller=SkillController(),
    )

    async def run():
        await env.reset(seed=5)
        step = await env.step(SKILL_ACTIONS.index("route_progression"))
        await env.close()
        return step

    step = asyncio.run(run())

    assert step.info["route_outcome"]["attempted"]
    assert step.info["pre_required_kill_route_penalty"] == 0.0
    assert step.info["route_action_reward"] == 0.0


def test_doom_agent_env_shapes_exit_ready_handoff_without_forcing_skill():
    def exit_ready_state(tick):
        state = _state(tick=tick, kills=1)
        exit_line = SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=60 * 65536, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=60 * 65536, y_fp=32 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=60 * 65536, y_fp=-32 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=60 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=60 * 65536,
            nearest_distance_fp=60 * 65536,
        )
        state.navigation.use_lines.append(exit_line)
        state.navigation.route_waypoint = SimpleNamespace(
            line=exit_line,
            priority=0,
            exit=True,
            walk_trigger=False,
        )
        return state

    async def run(action_name):
        first = exit_ready_state(1)
        second = exit_ready_state(2)
        client = _DurationAwareFakeClient([first, second])
        controller = SkillController()
        controller.policy._start_kills = 0
        env = DoomAgentEnv(
            DoomEnvConfig(
                max_steps=10,
                goal_preset="custom",
                max_action_tics=1,
                route_progress_reward=0.0,
                route_reached_reward=0.0,
                route_failure_penalty=0.0,
                exit_route_progress_reward=0.0,
                exit_route_reached_reward=0.0,
                exit_route_failure_penalty=0.0,
                exit_ready_press_reward=1.25,
                exit_ready_route_penalty=0.6,
            ),
            client=client,
            controller=controller,
        )
        await env.reset(seed=5)
        controller.policy._start_kills = 0
        mask = dict(zip(SKILL_ACTIONS, env.action_mask()))
        step = await env.step(SKILL_ACTIONS.index(action_name))
        await env.close()
        return mask, step

    route_mask, route_step = asyncio.run(run("route_progression"))
    press_mask, press_step = asyncio.run(run("press_exit"))

    assert route_mask["route_progression"]
    assert route_mask["press_exit"]
    assert press_mask["route_progression"]
    assert press_mask["press_exit"]
    assert route_step.info["skill"] == "route_progression"
    assert route_step.info["exit_ready_press_available"] is True
    assert route_step.info["exit_ready_switch_attempt"] is True
    assert route_step.info["exit_ready_action_reward"] == pytest.approx(-0.6)
    assert route_step.reward == pytest.approx(-0.6)
    assert press_step.info["skill"] == "press_exit"
    assert press_step.info["exit_ready_press_available"] is True
    assert press_step.info["exit_ready_switch_attempt"] is True
    assert press_step.info["exit_ready_action_reward"] == pytest.approx(1.25)
    assert press_step.reward == pytest.approx(1.25)


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
    assert step.info["route_action_reward"] == pytest.approx(1.0)
    assert features["prev_route_progression"] == 1.0
    assert features["prev_route_progress_norm"] == pytest.approx(0.5)
    assert features["route_waypoint_failed_recently"] == 0.0


def test_doom_agent_env_boosts_exit_route_outcome_reward():
    first = _state(tick=1, route=True, route_exit=True, x_units=0)
    second = _state(tick=2, route=True, route_exit=True, x_units=512)
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

    assert step.info["route_outcome"]["attempted"]
    assert step.info["route_outcome"]["exit"]
    assert step.info["route_outcome"]["reached"]
    assert step.info["route_action_reward"] == pytest.approx(2.75)
    assert step.reward >= step.info["route_action_reward"]


def test_route_outcome_uses_decision_line_and_preserves_waypoint_exit():
    previous = _state(tick=1, route=True, route_exit=True, x_units=0)
    current = _state(tick=2, route=True, route_exit=True, x_units=256)
    line = previous.navigation.route_waypoint.line
    line.line_id = 330
    line.special = 88
    previous.navigation.route_waypoint.exit = True
    previous.navigation.route_waypoint.walk_trigger = False

    outcome = _route_outcome(
        "route_progression",
        previous,
        current,
        decision={
            "skill": "route_to_progression_line",
            "use_line": {
                "line_id": 330,
                "special": 88,
                "distance": 512.0,
                "angle_delta": 0.0,
            },
        },
    )

    assert outcome["attempted"]
    assert outcome["line_id"] == 330
    assert outcome["exit"] is True
    assert outcome["walk_trigger"] is False
    assert outcome["target_source"] == "route_waypoint"


def test_route_outcome_marks_unmatched_exit_special_as_decision_line():
    previous = _state(tick=1, route=True, route_exit=False, x_units=0)
    current = _state(tick=2, route=True, route_exit=False, x_units=128)
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=512 * 65536, y_fp=128 * 65536, z_fp=0),
        start=SimpleNamespace(x_fp=512 * 65536, y_fp=64 * 65536, z_fp=0),
        end=SimpleNamespace(x_fp=512 * 65536, y_fp=192 * 65536, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=512 * 65536, y_fp=128 * 65536, z_fp=0),
        special=11,
        tag=0,
        distance_fp=512 * 65536,
        nearest_distance_fp=512 * 65536,
    )
    previous.navigation.use_lines.append(exit_line)

    outcome = _route_outcome(
        "route_progression",
        previous,
        current,
        decision={
            "skill": "route_to_progression_line",
            "use_line": {
                "line_id": 330,
                "special": 11,
                "distance": 512.0,
                "angle_delta": 0.0,
            },
        },
    )

    assert outcome["attempted"]
    assert outcome["line_id"] == 330
    assert outcome["exit"] is True
    assert outcome["target_source"] == "decision_line"


def test_route_outcome_counts_exit_recovery_decision_line():
    previous = _state(tick=1, route=True, route_exit=False, x_units=0)
    current = _state(tick=2, route=True, route_exit=False, x_units=128)
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=512 * 65536, y_fp=0, z_fp=0),
        start=SimpleNamespace(x_fp=512 * 65536, y_fp=-64 * 65536, z_fp=0),
        end=SimpleNamespace(x_fp=512 * 65536, y_fp=64 * 65536, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=512 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=512 * 65536,
        nearest_distance_fp=512 * 65536,
    )
    previous.navigation.use_lines.append(exit_line)

    outcome = _route_outcome(
        "recover_stuck",
        previous,
        current,
        decision={
            "skill": "unstick_route_to_exit_line",
            "use_line": {
                "line_id": 330,
                "special": 11,
                "distance": 512.0,
                "angle_delta": 0.0,
            },
        },
    )

    assert outcome["attempted"]
    assert outcome["line_id"] == 330
    assert outcome["exit"] is True
    assert outcome["target_source"] == "decision_line"
    assert outcome["progress_units"] == pytest.approx(128.0)


def test_route_outcome_counts_waypoint_recovery_decision_line():
    previous = _state(tick=1, route=True, route_exit=False, x_units=0)
    current = _state(tick=2, route=True, route_exit=False, x_units=128)
    line = previous.navigation.route_waypoint.line

    for decision_skill in (
        "unstick_route_to_waypoint_line",
        "unstick_backtrack_from_waypoint_line",
    ):
        outcome = _route_outcome(
            "recover_stuck",
            previous,
            current,
            decision={
                "skill": decision_skill,
                "use_line": {
                    "line_id": line.line_id,
                    "special": line.special,
                    "distance": 512.0,
                    "angle_delta": 0.0,
                },
            },
        )

        assert outcome["attempted"]
        assert outcome["line_id"] == line.line_id
        assert outcome["exit"] is False
        assert outcome["walk_trigger"] is True
        assert outcome["target_source"] == "route_waypoint"
        assert outcome["progress_units"] == pytest.approx(128.0)


def test_route_outcome_does_not_count_non_exit_recovery_decision_line():
    previous = _state(tick=1, route=True, route_exit=False, x_units=0)
    current = _state(tick=2, route=True, route_exit=False, x_units=128)
    manual_line = previous.navigation.route_waypoint.line
    manual_line.line_id = 151
    manual_line.special = 1

    outcome = _route_outcome(
        "recover_stuck",
        previous,
        current,
        decision={
            "skill": "unstick_route_to_exit_line",
            "use_line": {
                "line_id": 151,
                "special": 1,
                "distance": 512.0,
                "angle_delta": 0.0,
            },
        },
    )

    assert not outcome["attempted"]
    assert outcome["line_id"] == 151
    assert outcome["exit"] is False


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
    def __init__(self, states, *, seed_applied=False):
        self.states = list(states)
        self.seed_applied = bool(seed_applied)
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
            seed_applied=self.seed_applied,
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


class _StreamingFakeClient(_FakeClient):
    async def session(self, actions):
        async for action in actions:
            self.actions.append(action)
            break
        while self.states:
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
    route_exit=False,
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
            exit=route_exit,
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
            level_time=5 if level_time is None else level_time,
            total_kills=1,
            total_items=0,
            total_secrets=0,
            gamestate=0,
        ),
        has_delta_state=False,
        navigation=navigation,
        combat=combat_probe,
    )


def test_doom_agent_env_step_time_penalty_reduces_reward_per_macro_step():
    # ponytail: same fire macro-step run twice; only step_time_penalty differs.
    def run(step_time_penalty):
        first = _state(tick=1, kills=0, enemy=True, combat=True)
        second = _state(tick=2, kills=1)
        client = _FakeClient([first, second])
        env = DoomAgentEnv(
            DoomEnvConfig(
                max_steps=2,
                goal_preset="combat",
                step_time_penalty=step_time_penalty,
            ),
            client=client,
            controller=SkillController(),
        )

        async def _run():
            await env.reset(seed=99)
            step = await env.step(SKILL_ACTIONS.index("fire"))
            await env.close()
            return step

        return asyncio.run(_run())

    baseline = run(0.0)
    penalized = run(0.25)

    # Default (0.0) leaves reward unchanged.
    assert penalized.reward == pytest.approx(baseline.reward - 0.25)
