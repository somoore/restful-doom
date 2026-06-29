import asyncio
import json
from types import SimpleNamespace

from restfuldoom_agent.brain import (
    AgentMemory,
    BrainPolicy,
    BrainPolicyParams,
    EpisodeStats,
    LINE_ATTEMPT_STALL_TICS,
    cell_key,
    combat_from_state,
    doom_line_side,
    extract_features,
    navigation_from_state,
    normalize_angle_delta,
    turn_action_for_delta,
)
from restfuldoom_agent.client import agent_pb2


def state(*, tick=1, x=0, y=0, angle=0, enemy=None, combat=None, direction_probes=None):
    position = SimpleNamespace(x_fp=int(x * 65536), y_fp=int(y * 65536), z_fp=0)
    obj = SimpleNamespace(position=position, angle_degrees=angle)
    ammo = SimpleNamespace(bullets=50)
    player = SimpleNamespace(
        object=obj,
        health=100,
        ammo=ammo,
        kills=0,
        items=0,
        secrets=0,
    )
    enemies = []
    if enemy:
        enemy_position = SimpleNamespace(
            x_fp=int(enemy["x"] * 65536),
            y_fp=int(enemy["y"] * 65536),
            z_fp=0,
        )
        enemy_obj = SimpleNamespace(
            id=enemy.get("id", 7),
            position=enemy_position,
            distance_fp=int(enemy["distance"] * 65536),
            health=enemy.get("health", 20),
        )
        enemies.append(
            SimpleNamespace(
                object=enemy_obj,
                line_of_sight=enemy.get("line_of_sight", True),
            )
        )
    level = SimpleNamespace(episode=1, map=1, level_time=tick)
    navigation = SimpleNamespace(
        forward_open=True,
        back_open=True,
        left_open=True,
        right_open=True,
        use_line_ahead=False,
        front_blocking_line_special=0,
        front_block_distance_fp=96 * 65536,
        probe_distance_fp=96 * 65536,
    )
    if direction_probes is not None:
        navigation.direction_probes = [
            SimpleNamespace(
                angle_offset_degrees=probe["angle_offset_degrees"],
                open=probe.get("open", True),
                block_distance_fp=int(probe.get("block_distance", 96) * 65536),
                blocking_line_special=probe.get("blocking_line_special", 0),
                use_line_ahead=probe.get("use_line_ahead", False),
            )
            for probe in direction_probes
        ]
    if combat is None:
        combat = SimpleNamespace(
            has_shootable_target=False,
            target_id=0,
            target_health=0,
            target_distance_fp=0,
            aim_slope_fp=0,
            range_fp=0,
            target_is_enemy=False,
        )
    return SimpleNamespace(
        tick=tick,
        player=player,
        enemies=enemies,
        level=level,
        navigation=navigation,
        combat=combat,
    )


def test_angle_delta_normalization():
    assert normalize_angle_delta(370) == 10
    assert normalize_angle_delta(-190) == 170


def test_turn_action_reduces_positive_angle_delta():
    assert turn_action_for_delta(30) == 3
    assert turn_action_for_delta(-30) == 4


def test_cell_key_uses_stable_grid():
    assert cell_key(0, 0) == "0:0"
    assert cell_key(129, -1) == "1:-1"


def test_doom_line_side_matches_front_side_convention():
    line = {
        "start_x_units": -10,
        "start_y_units": 0,
        "end_x_units": 10,
        "end_y_units": 0,
    }

    assert doom_line_side(0, -5, line) == 0
    assert doom_line_side(0, 5, line) == 1


def test_extract_features_sorts_visible_enemy_by_threat(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")

    features = extract_features(
        state(enemy={"x": 100, "y": 0, "distance": 100}),
        memory,
        BrainPolicyParams(),
    )

    assert features.visible_enemies[0]["id"] == 7
    assert features.known_enemies[0]["id"] == 7
    assert features.visible_enemies[0]["angle_delta"] == 0
    assert features.visible_enemies[0]["threat"] > 0
    assert features.episode == 1
    assert features.map == 1
    assert features.navigation["forward_open"] is True


def test_navigation_defaults_are_conservative():
    nav = navigation_from_state(None)

    assert nav["forward_open"] is True
    assert nav["use_line_ahead"] is False
    assert nav["topology_frontier_count"] == 0


def test_extract_features_counts_low_visit_direction_frontiers(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    game_state = state(
        direction_probes=[
            {"angle_offset_degrees": 0, "open": True},
            {"angle_offset_degrees": 90, "open": True},
            {"angle_offset_degrees": -90, "open": True},
        ]
    )

    open_frontier = extract_features(game_state, memory, BrainPolicyParams())
    memory.data["cells"] = {
        "1:0": {"visits": 2},
        "0:1": {"visits": 2},
        "0:-1": {"visits": 2},
    }
    exhausted = extract_features(game_state, memory, BrainPolicyParams())

    assert open_frontier.navigation["topology_frontier_count"] == 3
    assert exhausted.navigation["topology_frontier_count"] == 0


def test_combat_defaults_are_conservative():
    combat = combat_from_state(None)

    assert combat["has_shootable_target"] is False
    assert combat["target_is_enemy"] is False


def test_episode_success_requires_kill_and_level_completion():
    stats = EpisodeStats("run", "candidate", "policy", "combat")
    stats.start_kills = 0
    stats.end_kills = 1
    stats.level_completed = True

    assert stats.succeeded(required_kills=1, require_level_complete=True)
    assert not stats.succeeded(required_kills=2, require_level_complete=True)


def test_episode_success_preserves_peak_kills_across_map_reset():
    stats = EpisodeStats("run", "candidate", "policy", "combat")
    stats.start_kills = 0
    stats.peak_kills = 6
    stats.end_kills = 0
    stats.start_items = 0
    stats.peak_items = 3
    stats.end_items = 0
    stats.level_completed = True

    assert stats.kill_delta() == 6
    assert stats.item_delta() == 3
    assert stats.summary()["kill_delta"] == 6
    assert stats.succeeded(required_kills=1, require_level_complete=True)


def test_best_params_merges_older_memory_files(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.agent_memory.v1",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "policy": {
                    "best_score": 1.0,
                    "best_params": {"move_amount": 33},
                    "generations": 1,
                },
                "cells": {},
                "enemies": {},
                "episodes": [],
                "lessons": [],
            }
        )
    )

    params = AgentMemory.load(path).best_params()

    assert params.move_amount == 33
    assert params.aim_tolerance_degrees == BrainPolicyParams().aim_tolerance_degrees


def test_policy_seeks_bounded_nonvisible_enemy_before_first_kill(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=40,
                enemy={
                    "x": 1200,
                    "y": -1200,
                    "distance": 1800,
                    "line_of_sight": False,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "seek_known_enemy"
    assert action.action != 0 or action.raw.forward_move > 0


def test_policy_does_not_crash_after_turn_decision(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    first = state(tick=1, x=0, y=0)
    second = state(tick=20, x=0, y=0)
    asyncio.run(policy.next_action(first))
    policy.last_decision = {"skill": "turn_from_block"}

    asyncio.run(policy.next_action(second))

    assert policy.last_decision["skill"] in {
        "break_cell_loop",
        "explore_frontier",
        "sweep_frontier",
    }


def test_policy_aims_with_correct_turn_direction(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "x": 300,
                    "y": 300,
                    "distance": 424,
                    "line_of_sight": True,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "aim_at_enemy"
    assert action.action == 3


def test_policy_holds_attack_when_visible_and_aligned(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "x": 400,
                    "y": 10,
                    "distance": 400,
                    "line_of_sight": True,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "fire_on_enemy"
    assert action.raw.buttons & 1


def test_policy_fires_when_combat_probe_has_enemy(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "id": 77,
                    "x": 500,
                    "y": 120,
                    "distance": 514,
                    "line_of_sight": True,
                },
                combat=SimpleNamespace(
                    has_shootable_target=True,
                    target_id=77,
                    target_health=20,
                    target_distance_fp=514 * 65536,
                    aim_slope_fp=0,
                    range_fp=2048 * 65536,
                    target_is_enemy=True,
                ),
            )
        )
    )

    assert policy.last_decision["skill"] == "fire_on_shootable_target"
    assert action.raw.buttons & 1


def test_critical_health_fires_while_retreating_when_aligned(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(
        tick=40,
        enemy={
            "id": 77,
            "x": 160,
            "y": 0,
            "distance": 160,
            "line_of_sight": True,
        },
    )
    game_state.player.health = 9

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "critical_defensive_fire"
    assert action.raw.buttons & 1
    assert action.raw.forward_move < 0


def test_policy_closes_far_visible_enemy_before_firing(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "x": 2200,
                    "y": 10,
                    "distance": 2200,
                    "line_of_sight": True,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "close_visible_contact"
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_policy_closes_far_angled_visible_enemy_when_front_is_open(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "x": 1600,
                    "y": 1600,
                    "distance": 2263,
                    "line_of_sight": True,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "close_visible_contact"
    assert action.raw.forward_move > 0
    assert action.raw.angle_turn > 0


def test_policy_closes_visible_contact_on_best_local_ray(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    action = asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "x": 1600,
                    "y": -1500,
                    "distance": 2193,
                    "line_of_sight": True,
                },
                direction_probes=[
                    {"angle_offset_degrees": 0, "open": True, "block_distance": 96},
                    {"angle_offset_degrees": -45, "open": False, "block_distance": 96},
                    {"angle_offset_degrees": 45, "open": True, "block_distance": 768},
                ],
            )
        )
    )

    assert policy.last_decision["skill"] == "close_visible_contact"
    assert policy.last_decision["direction_probe"]["angle_offset_degrees"] == -45
    assert action.raw.forward_move > 0
    assert action.raw.side_move > 0


def test_policy_does_not_hunt_stale_nonvisible_enemy_after_contact(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "id": 9,
                    "x": 600,
                    "y": 0,
                    "distance": 600,
                    "line_of_sight": True,
                },
            )
        )
    )
    action = asyncio.run(
        policy.next_action(
            state(
                tick=120,
                enemy={
                    "id": 9,
                    "x": 600,
                    "y": 0,
                    "distance": 600,
                    "line_of_sight": False,
                },
            )
        )
    )

    assert policy.last_decision["skill"] != "hunt_known_enemy"
    assert action.action != 0 or action.raw.forward_move > 0


def test_policy_pursues_recent_visible_contact_corridor(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    asyncio.run(
        policy.next_action(
            state(
                tick=5,
                enemy={
                    "id": 11,
                    "x": 2300,
                    "y": 0,
                    "distance": 2300,
                    "line_of_sight": True,
                },
            )
        )
    )
    action = asyncio.run(
        policy.next_action(
            state(
                tick=20,
                enemy={
                    "id": 11,
                    "x": 2300,
                    "y": 0,
                    "distance": 2300,
                    "line_of_sight": False,
                },
            )
        )
    )

    assert policy.last_decision["skill"] == "pursue_last_contact_corridor"
    assert action.raw.forward_move > 0


def test_policy_turns_toward_directional_use_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=False,
            block_distance_fp=64 * 65536,
            blocking_line_special=1,
            use_line_ahead=True,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "turn_to_use_line"
    assert action.action == 3


def test_policy_uses_aligned_nearby_special_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40, x=8)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=99,
            midpoint=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=3,
            distance_fp=64 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "use_nearby_line"
    assert action.raw.buttons & 2


def test_policy_uses_aligned_line_at_observed_stuck_distance(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40, x=8)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=151,
            midpoint=SimpleNamespace(x_fp=147 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=147 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=147 * 65536,
            nearest_distance_fp=147 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "use_nearby_line"
    assert action.raw.buttons & 2


def test_stuck_policy_uses_manual_line_ahead_before_unstick(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.navigation.forward_open = False
    game_state.navigation.left_open = False
    game_state.navigation.right_open = False
    game_state.navigation.use_line_ahead = True
    game_state.navigation.front_blocking_line_special = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=340,
            midpoint=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=16 * 65536,
            nearest_distance_fp=16 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "use_nearby_line"
    assert policy.last_decision["use_line"]["line_id"] == 340
    assert action.raw.buttons & 2


def test_policy_approaches_aligned_far_special_line_before_use(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=99,
            midpoint=SimpleNamespace(x_fp=256 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=256 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=3,
            distance_fp=256 * 65536,
            nearest_distance_fp=256 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "approach_nearby_use_line"
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_policy_ignores_non_manual_special_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=357,
            midpoint=SimpleNamespace(x_fp=392 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=392 * 65536, y_fp=0, z_fp=0),
            special=48,
            tag=0,
            distance_fp=392 * 65536,
            nearest_distance_fp=392 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] not in {
        "approach_nearby_use_line",
        "turn_to_nearby_use_line",
        "use_nearby_line",
    }
    assert action.raw.buttons == 0


def test_policy_escapes_remembered_hazard_cell_before_use_ray(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["cells"]["0:0"] = {
        "visits": 12,
        "damage_events": 1,
        "last_seen_tick": 1,
    }
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.player.health = 22
    game_state.navigation.use_line_ahead = True

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "escape_hazard_cell"
    assert action.raw.buttons == 0
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_policy_escapes_hazard_toward_progression_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["cells"]["0:0"] = {
        "visits": 12,
        "damage_events": 1,
        "last_seen_tick": 1,
    }
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.health = 22
    game_state.player.kills = 1
    game_state.navigation.use_line_ahead = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=50 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=50 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=50 * 65536,
            nearest_distance_fp=50 * 65536,
        ),
        SimpleNamespace(
            line_id=308,
            midpoint=SimpleNamespace(x_fp=480 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=480 * 65536, y_fp=0, z_fp=0),
            special=36,
            tag=1,
            distance_fp=480 * 65536,
            nearest_distance_fp=480 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "escape_hazard_toward_progression"
    assert policy.last_decision["use_line"]["line_id"] == 308
    assert action.raw.forward_move > 0
    assert abs(action.raw.side_move) < action.raw.forward_move


def test_low_health_hazard_prioritizes_visible_exit_over_walk_trigger(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["cells"]["0:0"] = {
        "visits": 12,
        "damage_events": 1,
        "last_seen_tick": 1,
    }
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.health = 12
    game_state.player.kills = 6
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=800 * 65536,
            nearest_distance_fp=800 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 330
    assert policy.last_decision["skill"] == "approach_progression_line"
    assert action.raw.forward_move > 0


def test_critical_health_hazard_escapes_instead_of_chasing_walk_trigger(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["cells"]["0:0"] = {
        "visits": 12,
        "damage_events": 1,
        "last_seen_tick": 1,
    }
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.health = 7
    game_state.player.kills = 6
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        )
    ]

    asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "escape_hazard_cell"
    assert policy.last_decision["direction_probe"]["angle_offset_degrees"] == 0


def test_post_kill_policy_targets_progression_trigger_before_nearby_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=151,
            midpoint=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=64 * 65536,
            nearest_distance_fp=64 * 65536,
        ),
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "cross_progression_line"
    assert policy.last_decision["use_line"]["line_id"] == 195
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_post_kill_policy_skips_far_exit_until_route_is_clearer(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=1000 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1000 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=1000 * 65536,
            nearest_distance_fp=1000 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1800 * 65536,
            nearest_distance_fp=1800 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 195
    assert policy.last_decision["skill"] == "cross_progression_line"
    assert action.raw.forward_move > 0


def test_post_combat_policy_prefers_visible_far_exit_when_combat_is_quiet(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 5
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=1000 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1000 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=1000 * 65536,
            nearest_distance_fp=1000 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1800 * 65536,
            nearest_distance_fp=1800 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 330
    assert policy.last_decision["skill"] == "approach_progression_line"
    assert action.raw.forward_move > 0


def test_post_combat_snapshot_prefers_visible_exit_with_inherited_kills(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=1600 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1600 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=1600 * 65536,
            nearest_distance_fp=1600 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2944 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2944 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2944 * 65536,
            nearest_distance_fp=2944 * 65536,
        ),
    ]
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5

    selected = policy._select_progression_line(features)
    action, decision = policy._advance_progression_line(features, selected, stuck=False)

    assert selected["line_id"] == 330
    assert decision["use_line"]["line_id"] == 330
    assert decision["skill"] == "approach_progression_line"
    assert action.raw.forward_move > 0


def test_visible_post_combat_exit_route_keeps_remembered_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    params = BrainPolicyParams()
    policy = BrainPolicy(
        memory=memory,
        params=params,
        policy_id="test-policy",
    )
    policy._start_kills = 5
    policy._last_post_combat_exit_line_id = 330
    policy._last_post_combat_exit_tick = 32

    game_state = state(
        tick=40,
        enemy={"x": 512, "y": 128, "distance": 528},
    )
    game_state.player.kills = 5
    game_state.navigation.forward_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=324,
            midpoint=SimpleNamespace(x_fp=372 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=372 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=372 * 65536,
            nearest_distance_fp=372 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        ),
    ]

    features = extract_features(game_state, memory, params)
    line = policy._select_progression_line(features)

    assert line is not None
    assert line["line_id"] == 330


def test_visible_post_combat_exit_route_bypasses_forward_open_assist_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    params = BrainPolicyParams()
    policy = BrainPolicy(
        memory=memory,
        params=params,
        policy_id="test-policy",
    )
    policy._start_kills = 5
    policy._last_post_combat_exit_line_id = 330
    policy._last_post_combat_exit_tick = 32

    game_state = state(
        tick=40,
        enemy={"x": 512, "y": 128, "distance": 528},
    )
    game_state.player.kills = 5
    game_state.navigation.forward_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=324,
            midpoint=SimpleNamespace(x_fp=372 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=372 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=372 * 65536,
            nearest_distance_fp=372 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        ),
    ]

    features = extract_features(game_state, memory, params)
    line = policy._select_progression_line(features)
    assert line is not None

    action, decision = policy._advance_progression_line(features, line, stuck=False)

    assert decision["use_line"]["line_id"] == 330
    assert decision["skill"] == "approach_progression_line"
    assert action.raw.forward_move > 0


def test_post_combat_snapshot_finishes_nearby_walk_route_waypoint_before_far_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=2250 * 65536,
        nearest_distance_fp=2250 * 65536,
    )
    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.use_lines = [route_line, exit_line]
    game_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5

    selected = policy._select_progression_line(features)
    action, decision = policy._advance_progression_line(features, selected, stuck=False)

    assert selected["line_id"] == 195
    assert decision["use_line"]["line_id"] == 195
    assert decision["skill"] == "cross_progression_line"
    assert action.raw.forward_move > 0


def test_stalled_nearby_walk_route_yields_to_visible_far_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=2250 * 65536,
        nearest_distance_fp=2250 * 65536,
    )

    first_state = state(tick=40)
    first_state.player.kills = 5
    first_state.navigation.use_lines = [route_line, exit_line]
    first_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    policy._start_kills = 5
    first_features = extract_features(first_state, policy.memory, policy.params)
    selected = policy._select_progression_line(first_features)

    assert selected["line_id"] == 195
    assert policy._record_line_attempt(first_features, selected) is True

    stalled_state = state(tick=40 + LINE_ATTEMPT_STALL_TICS + 1)
    stalled_state.player.kills = 5
    stalled_state.navigation.use_lines = [route_line, exit_line]
    stalled_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    stalled_features = extract_features(stalled_state, policy.memory, policy.params)
    stalled_selection = policy._select_progression_line(stalled_features)

    assert stalled_selection["line_id"] == 195
    assert policy._record_line_attempt(stalled_features, stalled_selection) is False

    selected_after_stall = policy._select_progression_line(stalled_features)

    assert selected_after_stall["line_id"] == 330


def test_walk_route_stall_across_cells_yields_to_visible_far_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=2250 * 65536,
        nearest_distance_fp=2250 * 65536,
    )

    first_state = state(tick=40, x=0, y=0)
    first_state.player.kills = 5
    first_state.navigation.use_lines = [route_line, exit_line]
    first_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    policy._start_kills = 5
    first_features = extract_features(first_state, policy.memory, policy.params)
    selected = policy._select_progression_line(first_features)

    assert selected["line_id"] == 195
    assert policy._record_line_attempt(first_features, selected) is True

    stalled_state = state(tick=40 + LINE_ATTEMPT_STALL_TICS + 1, x=0, y=128)
    stalled_state.player.kills = 5
    stalled_state.navigation.use_lines = [route_line, exit_line]
    stalled_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    stalled_features = extract_features(stalled_state, policy.memory, policy.params)
    stalled_selection = policy._select_progression_line(stalled_features)

    assert stalled_features.cell != first_features.cell
    assert stalled_selection["line_id"] == 195
    assert policy._record_line_attempt(stalled_features, stalled_selection) is False

    selected_after_stall = policy._select_progression_line(stalled_features)

    assert selected_after_stall["line_id"] == 330


def test_walk_route_stall_before_kill_remains_cell_scoped(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )

    first_state = state(tick=40, x=0, y=0)
    first_state.navigation.use_lines = [route_line]
    first_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    first_features = extract_features(first_state, policy.memory, policy.params)
    first_line = first_features.navigation["route_waypoint"]["line"]

    assert policy._record_line_attempt(first_features, first_line) is True

    stalled_state = state(tick=40 + LINE_ATTEMPT_STALL_TICS + 1, x=0, y=128)
    stalled_state.navigation.use_lines = [route_line]
    stalled_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    stalled_features = extract_features(stalled_state, policy.memory, policy.params)
    stalled_line = stalled_features.navigation["route_waypoint"]["line"]

    assert stalled_features.cell != first_features.cell
    assert policy._record_line_attempt(stalled_features, stalled_line) is True
    assert not policy._is_line_blocked(stalled_features, stalled_line)


def test_reached_walk_route_waypoint_yields_to_visible_far_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=24 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=24 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=24 * 65536,
        nearest_distance_fp=24 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=1800 * 65536,
        nearest_distance_fp=1800 * 65536,
    )
    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.use_lines = [route_line, exit_line]
    game_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    policy._start_kills = 5
    features = extract_features(game_state, policy.memory, policy.params)

    selected = policy._select_progression_line(features)

    assert selected["line_id"] == 330
    assert 195 in policy._completed_walk_route_line_ids


def test_post_combat_stuck_recovery_finishes_nearby_walk_route_waypoint(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=2250 * 65536,
        nearest_distance_fp=2250 * 65536,
    )
    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=128 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [route_line, exit_line]
    game_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5

    action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] == "unstick_route_to_waypoint_line"
    assert decision["use_line"]["line_id"] == 195
    assert decision["direction_probe"]["angle_offset_degrees"] == 30
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_blocked_post_combat_route_progression_backtracks_from_nearby_waypoint(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    route_line = SimpleNamespace(
        line_id=195,
        midpoint=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=560 * 65536, y_fp=0, z_fp=0),
        special=88,
        tag=2,
        distance_fp=560 * 65536,
        nearest_distance_fp=560 * 65536,
    )
    exit_line = SimpleNamespace(
        line_id=330,
        midpoint=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        nearest_point=SimpleNamespace(x_fp=2250 * 65536, y_fp=0, z_fp=0),
        special=11,
        tag=0,
        distance_fp=2250 * 65536,
        nearest_distance_fp=2250 * 65536,
    )
    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.forward_open = False
    game_state.navigation.left_open = False
    game_state.navigation.right_open = False
    game_state.navigation.back_open = True
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=offset,
            open=False,
            block_distance_fp=96 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
        for offset in (-90, -60, -30, -15, 0, 15, 30, 60, 90)
    ]
    game_state.navigation.use_lines = [route_line, exit_line]
    game_state.navigation.route_waypoint = SimpleNamespace(
        line=route_line,
        priority=0,
        exit=False,
        walk_trigger=True,
    )
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5

    selected = policy._select_progression_line(features)
    action, decision = policy._advance_progression_line(features, selected, stuck=False)

    assert selected["line_id"] == 195
    assert decision["skill"] == "unstick_backtrack_from_waypoint_line"
    assert decision["use_line"]["line_id"] == 195
    assert action.raw.forward_move < 0


def test_far_exit_preference_does_not_preempt_visible_combat(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(
        tick=40,
        x=8,
        enemy={"x": 256, "y": 0, "distance": 256, "line_of_sight": True},
    )
    game_state.player.kills = 3
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=1800 * 65536,
            nearest_distance_fp=1800 * 65536,
        ),
    ]

    asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] in {"engage_enemy", "fire_on_enemy"}
    assert policy.last_decision.get("use_line", {}).get("line_id") != 330


def test_post_kill_policy_delays_far_walk_trigger_to_hunt_remaining_enemy(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    policy._start_kills = 0

    game_state = state(
        tick=40,
        enemy={"x": 500, "y": 0, "distance": 500, "line_of_sight": False},
    )
    game_state.player.kills = 2
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=308,
            midpoint=SimpleNamespace(x_fp=1900 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1900 * 65536, y_fp=0, z_fp=0),
            special=36,
            tag=1,
            distance_fp=1900 * 65536,
            nearest_distance_fp=1900 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "seek_known_enemy"
    assert action.action == 1


def test_midrange_walk_trigger_waits_until_post_combat_kill_count(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    policy._start_kills = 0

    game_state = state(tick=40)
    game_state.player.kills = 4
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=308,
            midpoint=SimpleNamespace(x_fp=1100 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1100 * 65536, y_fp=0, z_fp=0),
            special=36,
            tag=1,
            distance_fp=1100 * 65536,
            nearest_distance_fp=1100 * 65536,
        )
    ]
    features = extract_features(game_state, policy.memory, policy.params)

    assert policy._select_progression_line(features) is None

    game_state.player.kills = 5
    ready_features = extract_features(game_state, policy.memory, policy.params)
    selected = policy._select_progression_line(ready_features)

    assert selected is not None
    assert selected["line_id"] == 308


def test_post_kill_progression_routes_on_open_ray_when_forward_blocked(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=0,
            open=False,
            block_distance_fp=16 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        ),
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=96 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        ),
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "route_to_progression_line"
    assert policy.last_decision["direction_probe"]["angle_offset_degrees"] == 30
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_stuck_post_kill_policy_keeps_routing_to_progression_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))
    policy._last_position = (0.0, 0.0)
    policy._last_progress_tick = 1
    policy.last_decision = {"skill": "route_to_progression_line"}

    game_state = state(tick=40, x=0, y=0)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=0,
            open=False,
            block_distance_fp=16 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        ),
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=96 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        ),
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=308,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=36,
            tag=1,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "route_to_progression_line"
    assert policy.last_decision["use_line"]["line_id"] == 308
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_blocked_far_exit_routes_before_turning_in_place(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=96 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=810 * 65536,
            nearest_distance_fp=810 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "route_to_progression_line"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert policy.last_decision["direction_probe"]["angle_offset_degrees"] == 30
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_stuck_recovery_routes_toward_visible_exit_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=128 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=810 * 65536,
            nearest_distance_fp=810 * 65536,
        )
    ]
    features = extract_features(game_state, policy.memory, policy.params)

    action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] == "unstick_route_to_exit_line"
    assert decision["use_line"]["line_id"] == 330
    assert decision["direction_probe"]["angle_offset_degrees"] == 30
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_stuck_exit_recovery_slides_after_assist_doors_stall(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    def use_line(
        *,
        line_id,
        special,
        midpoint,
        nearest,
        start,
        end,
        distance,
    ):
        return SimpleNamespace(
            line_id=line_id,
            midpoint=SimpleNamespace(
                x_fp=int(midpoint[0] * 65536),
                y_fp=int(midpoint[1] * 65536),
                z_fp=0,
            ),
            nearest_point=SimpleNamespace(
                x_fp=int(nearest[0] * 65536),
                y_fp=int(nearest[1] * 65536),
                z_fp=0,
            ),
            start=SimpleNamespace(
                x_fp=int(start[0] * 65536),
                y_fp=int(start[1] * 65536),
                z_fp=0,
            ),
            end=SimpleNamespace(
                x_fp=int(end[0] * 65536),
                y_fp=int(end[1] * 65536),
                z_fp=0,
            ),
            special=special,
            tag=0,
            distance_fp=int(distance * 65536),
            nearest_distance_fp=int(distance * 65536),
        )

    game_state = state(tick=100, x=2918, y=-4336, angle=270)
    game_state.player.kills = 6
    game_state.navigation.forward_open = True
    game_state.navigation.back_open = False
    game_state.navigation.use_lines = [
        use_line(
            line_id=324,
            special=1,
            midpoint=(3008, -4648),
            nearest=(2976, -4648),
            start=(2976, -4648),
            end=(3040, -4648),
            distance=341,
        ),
        use_line(
            line_id=325,
            special=1,
            midpoint=(3008, -4632),
            nearest=(2976, -4632),
            start=(3040, -4632),
            end=(2976, -4632),
            distance=325,
        ),
        use_line(
            line_id=330,
            special=11,
            midpoint=(2912, -4768),
            nearest=(2912, -4736),
            start=(2912, -4800),
            end=(2912, -4736),
            distance=403,
        ),
    ]
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5
    policy._last_progress_tick = features.tick - 16
    for line in features.navigation["use_lines"]:
        if line["line_id"] in {324, 325}:
            policy._blocked_use_lines[policy._line_key(features.cell, line)] = (
                features.tick + 120
            )

    action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] == "unstick_slide_to_exit_line"
    assert decision["use_line"]["line_id"] == 330
    assert decision["assist_line"]["line_id"] == 325
    assert action.raw.forward_move == 0
    assert action.raw.side_move < 0


def test_stuck_exit_recovery_backtracks_without_assist_slide(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    game_state = state(tick=100)
    game_state.player.kills = 6
    game_state.navigation.forward_open = True
    game_state.navigation.back_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=420 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=420 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=420 * 65536,
            nearest_distance_fp=420 * 65536,
        ),
    ]
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5
    policy._last_progress_tick = features.tick - 8

    action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] == "unstick_backtrack_from_exit_line"
    assert decision["stuck_phase"] == 1
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move < 0


def test_post_combat_snapshot_recovery_routes_toward_far_visible_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )

    game_state = state(tick=40)
    game_state.player.kills = 5
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=128 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=1600 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=1600 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=1600 * 65536,
            nearest_distance_fp=1600 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2944 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2944 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=2944 * 65536,
            nearest_distance_fp=2944 * 65536,
        ),
    ]
    features = extract_features(game_state, policy.memory, policy.params)
    policy._start_kills = 5

    action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] == "unstick_route_to_exit_line"
    assert decision["use_line"]["line_id"] == 330
    assert decision["direction_probe"]["angle_offset_degrees"] == 30
    assert action.raw.forward_move > 0
    assert action.raw.side_move != 0


def test_stuck_recovery_does_not_route_exit_during_visible_contact(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(
        tick=40,
        enemy={"x": 512, "y": 0, "distance": 512, "line_of_sight": True},
    )
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=30,
            open=True,
            block_distance_fp=128 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=760 * 65536, y_fp=280 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=810 * 65536,
            nearest_distance_fp=810 * 65536,
        )
    ]
    features = extract_features(game_state, policy.memory, policy.params)

    _action, decision = policy._recover_from_stuck(features)

    assert decision["skill"] != "unstick_route_to_exit_line"


def test_post_combat_exit_memory_suppresses_stale_walk_trigger_while_exit_visible(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    exit_state = state(tick=40)
    exit_state.player.kills = 5
    exit_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        ),
    ]

    asyncio.run(policy.next_action(exit_state))

    assert policy.last_decision["use_line"]["line_id"] == 330

    stale_state = state(tick=80)
    stale_state.player.kills = 5
    stale_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=3000 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=3000 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=3000 * 65536,
            nearest_distance_fp=3000 * 65536,
        )
    ]
    stale_features = extract_features(stale_state, policy.memory, policy.params)

    selected_while_exit_visible = policy._select_progression_line(stale_features)

    assert selected_while_exit_visible is not None
    assert selected_while_exit_visible["line_id"] == 330

    exit_gone_state = state(tick=90)
    exit_gone_state.player.kills = 5
    exit_gone_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=640 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=640 * 65536,
            nearest_distance_fp=640 * 65536,
        )
    ]
    exit_gone_features = extract_features(exit_gone_state, policy.memory, policy.params)

    selected_after_exit_gone = policy._select_progression_line(exit_gone_features)

    assert selected_after_exit_gone is not None
    assert selected_after_exit_gone["line_id"] == 195


def test_post_kill_policy_prioritizes_local_exit_line_but_approaches_until_close(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=195,
            midpoint=SimpleNamespace(x_fp=448 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=448 * 65536, y_fp=0, z_fp=0),
            special=88,
            tag=2,
            distance_fp=448 * 65536,
            nearest_distance_fp=448 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "approach_progression_line"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_post_kill_policy_uses_exit_line_at_close_range(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=80 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=80 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=80 * 65536,
            nearest_distance_fp=80 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "press_exit_switch"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2
    assert action.duration_tics == 1

    release_state = state(tick=41, x=8)
    release_state.player.kills = 1
    release_state.navigation.use_lines = game_state.navigation.use_lines
    release_action = asyncio.run(policy.next_action(release_state))

    assert policy.last_decision["skill"] == "release_exit_use"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert release_action.raw.buttons == 0
    assert release_action.duration_tics == 1


def test_close_exit_press_preempts_blocked_front_routing(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=-60,
            open=True,
            block_distance_fp=96 * 65536,
            blocking_line_special=0,
            use_line_ahead=False,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=16 * 65536,
            nearest_distance_fp=16 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "press_exit_switch"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2
    assert action.duration_tics == 1


def test_local_exit_opens_nearby_assist_door_when_blocked(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=64 * 65536,
            nearest_distance_fp=64 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "use_exit_assist_door"
    assert policy.last_decision["use_line"]["line_id"] == 325
    assert action.raw.buttons & 2
    assert action.raw.forward_move > 0


def test_exit_assist_ignores_manual_line_far_from_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 5
    game_state.navigation.forward_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=247,
            midpoint=SimpleNamespace(x_fp=(-80) * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=(-80) * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=88 * 65536,
            nearest_distance_fp=88 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=800 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=800 * 65536,
            nearest_distance_fp=800 * 65536,
        ),
    ]

    asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] != "use_exit_assist_door"
    assert policy.last_decision["use_line"]["line_id"] == 330


def test_local_exit_opens_immediate_manual_blocker_first(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.left_open = False
    game_state.navigation.right_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=340,
            midpoint=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=16 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=16 * 65536,
            nearest_distance_fp=16 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=760 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=760 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=760 * 65536,
            nearest_distance_fp=760 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 340
    assert action.raw.buttons & 2


def test_local_exit_prefers_assist_door_closest_to_exit(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=60 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=48 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=48 * 65536,
            nearest_distance_fp=48 * 65536,
        ),
        SimpleNamespace(
            line_id=324,
            midpoint=SimpleNamespace(x_fp=100 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=64 * 65536,
            nearest_distance_fp=64 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]

    asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 324


def test_far_local_exit_approaches_assist_door_even_when_forward_open(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, x=8)
    game_state.player.kills = 6
    game_state.navigation.forward_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=320 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=320 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=320 * 65536,
            nearest_distance_fp=320 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=400 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=400 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=400 * 65536,
            nearest_distance_fp=400 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 325
    assert policy.last_decision["skill"] == "approach_nearby_use_line"
    assert action.raw.forward_move > 0


def test_repeated_exit_assist_attempts_fall_back_to_exit_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    for tick in range(40, 92):
        game_state = state(tick=tick)
        game_state.player.kills = 6
        game_state.navigation.forward_open = False
        game_state.navigation.use_line_ahead = True
        game_state.navigation.use_lines = [
            SimpleNamespace(
                line_id=324,
                midpoint=SimpleNamespace(x_fp=100 * 65536, y_fp=0, z_fp=0),
                nearest_point=SimpleNamespace(x_fp=64 * 65536, y_fp=0, z_fp=0),
                special=1,
                tag=0,
                distance_fp=64 * 65536,
                nearest_distance_fp=64 * 65536,
            ),
            SimpleNamespace(
                line_id=330,
                midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
                nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
                special=11,
                tag=0,
                distance_fp=168 * 65536,
                nearest_distance_fp=168 * 65536,
            ),
        ]
        asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["use_line"]["line_id"] == 330
    assert policy.last_decision["skill"] in {"push_exit_switch", "release_exit_use"}


def test_local_exit_uses_close_angled_assist_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=64 * 65536, y_fp=54 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=64 * 65536, y_fp=54 * 65536, z_fp=0),
            special=1,
            tag=0,
            distance_fp=84 * 65536,
            nearest_distance_fp=84 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "turn_to_exit_assist_door"
    assert policy.last_decision["use_line"]["line_id"] == 325
    assert action.action in {3, 4}


def test_local_exit_approaches_assist_door_until_close(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=152 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=152 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=152 * 65536,
            nearest_distance_fp=152 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=280 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=280 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=280 * 65536,
            nearest_distance_fp=280 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "approach_nearby_use_line"
    assert policy.last_decision["use_line"]["line_id"] == 325
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_local_exit_presses_switch_near_wall_without_assist_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40)
    game_state.player.kills = 1
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = []
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "push_exit_switch"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2
    assert action.raw.forward_move > 0


def test_recent_exit_line_target_rebuilds_when_live_line_drops(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    first = state(tick=40, x=2961.9, y=-4727.1, angle=0)
    first.player.kills = 6
    first.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=2912 * 65536, y_fp=-4768 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=2912 * 65536, y_fp=-4736 * 65536, z_fp=0),
            start=SimpleNamespace(x_fp=2912 * 65536, y_fp=-4800 * 65536, z_fp=0),
            end=SimpleNamespace(x_fp=2912 * 65536, y_fp=-4736 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=60 * 65536,
            nearest_distance_fp=60 * 65536,
        )
    ]
    first_features = extract_features(first, memory, BrainPolicyParams())
    policy._remember_post_combat_exit_line(
        first_features,
        first_features.navigation["use_lines"][0],
    )

    later = state(tick=60, x=2961.9, y=-4727.1, angle=0)
    later.player.kills = 6
    later.navigation.use_lines = []
    later_features = extract_features(later, memory, BrainPolicyParams())

    target = policy._last_post_combat_exit_line_target(later_features)

    assert target is not None
    assert target["line_id"] == 330
    assert target["special"] == 11
    assert target["distance"] < 64
    assert "front_distance" in target


def test_close_exit_line_approaches_front_before_push_window(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=120 * 65536, y_fp=120 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=170 * 65536,
            nearest_distance_fp=170 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    line = features.navigation["use_lines"][0]
    line["angle_delta"] = 20.0
    line["front_angle_delta"] = 57.0
    line["front_distance"] = 152.0
    line["side"] = 0

    action, decision = policy._advance_progression_line(features, line, stuck=False)

    assert decision["skill"] == "approach_exit_switch_front"
    assert decision["use_line"]["line_id"] == 330
    assert not action.raw.buttons
    assert action.raw.forward_move > 0


def test_close_exit_line_pushes_before_side_manual_probe(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=90,
            open=False,
            block_distance_fp=48 * 65536,
            blocking_line_special=1,
            use_line_ahead=True,
        )
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "push_exit_switch"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2


def test_close_exit_line_pushes_before_front_manual_probe(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))
    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.direction_probes = [
        SimpleNamespace(
            angle_offset_degrees=0,
            open=False,
            block_distance_fp=64 * 65536,
            blocking_line_special=1,
            use_line_ahead=True,
        ),
        SimpleNamespace(
            angle_offset_degrees=30,
            open=False,
            block_distance_fp=72 * 65536,
            blocking_line_special=1,
            use_line_ahead=True,
        ),
    ]
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "push_exit_switch"
    assert policy.last_decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2


def test_repeated_close_exit_push_keeps_approaching_clear_front_point(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))
    first = state(tick=40)
    first.player.kills = 6
    first.navigation.forward_open = False
    first.navigation.back_open = True
    first.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]
    asyncio.run(policy.next_action(first))

    later = state(tick=100)
    later.player.kills = 6
    later.navigation.forward_open = False
    later.navigation.back_open = True
    later.navigation.use_lines = first.navigation.use_lines
    features = extract_features(later, memory, BrainPolicyParams())
    exit_line = features.navigation["use_lines"][0]
    exit_line["front_angle_delta"] = 37.0
    exit_line["front_distance"] = 152.0
    action, decision = policy._advance_progression_line(features, exit_line, stuck=False)

    assert decision["skill"] == "approach_exit_switch_front"
    assert decision["use_line"]["line_id"] == 330
    assert not action.raw.buttons
    assert action.raw.forward_move > 0


def test_stalled_exit_push_uses_front_blocker_before_recovery(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=100)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.back_open = True
    game_state.navigation.use_line_ahead = True
    game_state.navigation.front_blocking_line_special = 1
    game_state.navigation.front_block_distance_fp = 16 * 65536
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    exit_line = features.navigation["use_lines"][0]
    policy._exit_push_attempts[policy._line_key(features.cell, exit_line)] = {
        "first_tick": 40,
        "best_distance": 168.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }

    action, decision = policy._advance_progression_line(features, exit_line, stuck=False)

    assert decision["skill"] == "use_exit_route_blocker_ahead"
    assert action.action == agent_pb2.ACTION_USE


def test_stalled_exit_push_closes_from_front_side_when_front_point_is_near(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=100)
    game_state.player.kills = 6
    game_state.navigation.forward_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=128 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=112 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=112 * 65536,
            nearest_distance_fp=112 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    exit_line = features.navigation["use_lines"][0]
    exit_line["front_distance"] = 16.0
    exit_line["front_angle_delta"] = -55.0
    policy._exit_push_attempts[policy._line_key(features.cell, exit_line)] = {
        "first_tick": 40,
        "best_distance": 112.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }

    action, decision = policy._advance_progression_line(features, exit_line, stuck=False)

    assert decision["skill"] == "push_exit_switch"
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.buttons & 2
    assert action.raw.forward_move > 0


def test_stalled_exit_push_recovers_instead_of_retrying_assist_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=100)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.back_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=65 * 65536, y_fp=14 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=65 * 65536, y_fp=14 * 65536, z_fp=0),
            special=1,
            tag=0,
            distance_fp=66 * 65536,
            nearest_distance_fp=66 * 65536,
        ),
        SimpleNamespace(
            line_id=324,
            midpoint=SimpleNamespace(x_fp=82 * 65536, y_fp=18 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=82 * 65536, y_fp=18 * 65536, z_fp=0),
            special=1,
            tag=0,
            distance_fp=84 * 65536,
            nearest_distance_fp=84 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    exit_line = next(
        line for line in features.navigation["use_lines"] if line["line_id"] == 330
    )
    retry_line = next(
        line for line in features.navigation["use_lines"] if line["line_id"] == 325
    )
    policy._exit_push_attempts[policy._line_key(features.cell, exit_line)] = {
        "first_tick": 40,
        "best_distance": 168.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }
    policy._blocked_use_lines[policy._line_key(features.cell, retry_line)] = 999

    action, decision = policy._advance_progression_line(features, exit_line, stuck=False)

    assert decision["skill"] == "recover_exit_switch_approach"
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move < 0
    assert action.raw.side_move != 0


def test_stalled_exit_push_recovers_instead_of_turning_to_retry_assist_door(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=100)
    game_state.player.kills = 6
    game_state.navigation.forward_open = False
    game_state.navigation.back_open = True
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=325,
            midpoint=SimpleNamespace(x_fp=65 * 65536, y_fp=35 * 65536, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=65 * 65536, y_fp=35 * 65536, z_fp=0),
            special=1,
            tag=0,
            distance_fp=74 * 65536,
            nearest_distance_fp=74 * 65536,
        ),
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=168 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=168 * 65536,
            nearest_distance_fp=168 * 65536,
        ),
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    exit_line = next(
        line for line in features.navigation["use_lines"] if line["line_id"] == 330
    )
    policy._exit_push_attempts[policy._line_key(features.cell, exit_line)] = {
        "first_tick": 40,
        "best_distance": 168.0,
        "signature": {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        },
    }

    action, decision = policy._advance_progression_line(features, exit_line, stuck=False)

    assert decision["skill"] == "recover_exit_switch_approach"
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move < 0
    assert action.raw.side_move != 0


def test_close_exit_line_uses_front_approach_when_front_point_is_aligned(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.player.kills = 6
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=200 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=120 * 65536, y_fp=120 * 65536, z_fp=0),
            special=11,
            tag=0,
            distance_fp=170 * 65536,
            nearest_distance_fp=170 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    line = features.navigation["use_lines"][0]
    line["angle_delta"] = 45.0
    line["front_angle_delta"] = 0.0
    line["front_distance"] = 152.0
    line["side"] = 0

    action, decision = policy._advance_progression_line(features, line, stuck=False)

    assert decision["skill"] == "approach_exit_switch_front"
    assert decision["use_line"]["line_id"] == 330
    assert action.raw.forward_move > 0
    assert not action.raw.buttons


def test_policy_approaches_front_side_before_using_manual_line(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    asyncio.run(policy.next_action(state(tick=1)))

    game_state = state(tick=40, y=10, angle=270)
    game_state.player.kills = 1
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=0, y_fp=0, z_fp=0),
            start=SimpleNamespace(x_fp=-10 * 65536, y_fp=0, z_fp=0),
            end=SimpleNamespace(x_fp=10 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=0, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=10 * 65536,
            nearest_distance_fp=10 * 65536,
        )
    ]

    action = asyncio.run(policy.next_action(game_state))

    assert policy.last_decision["skill"] == "approach_exit_switch_front"
    assert policy.last_decision["use_line"]["side"] == 1
    assert action.raw.forward_move > 0
    assert action.raw.buttons == 0


def test_line_attempt_keeps_trying_while_distance_improves(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=500 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=500 * 65536,
            nearest_distance_fp=500 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    line = features.navigation["use_lines"][0]

    assert policy._record_line_attempt(features, line) is True

    later_state = state(tick=100)
    later_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=330,
            midpoint=SimpleNamespace(x_fp=460 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=460 * 65536, y_fp=0, z_fp=0),
            special=11,
            tag=0,
            distance_fp=460 * 65536,
            nearest_distance_fp=460 * 65536,
        )
    ]
    later_features = extract_features(later_state, memory, BrainPolicyParams())
    later_line = later_features.navigation["use_lines"][0]

    assert policy._record_line_attempt(later_features, later_line) is True
    assert policy._is_line_blocked(later_features, later_line) is False


def test_policy_blacklists_repeated_line_attempt_in_same_cell(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test-policy",
    )
    game_state = state(tick=40)
    game_state.navigation.use_lines = [
        SimpleNamespace(
            line_id=151,
            midpoint=SimpleNamespace(x_fp=256 * 65536, y_fp=0, z_fp=0),
            nearest_point=SimpleNamespace(x_fp=256 * 65536, y_fp=0, z_fp=0),
            special=1,
            tag=0,
            distance_fp=256 * 65536,
            nearest_distance_fp=256 * 65536,
        )
    ]
    features = extract_features(game_state, memory, BrainPolicyParams())
    line = features.navigation["use_lines"][0]

    assert policy._record_line_attempt(features, line) is True

    later_state = state(tick=101)
    later_state.navigation.use_lines = game_state.navigation.use_lines
    later_features = extract_features(later_state, memory, BrainPolicyParams())
    later_line = later_features.navigation["use_lines"][0]

    assert policy._record_line_attempt(later_features, later_line) is False
    assert policy._is_line_blocked(later_features, later_line) is True
