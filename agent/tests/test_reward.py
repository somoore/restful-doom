from types import SimpleNamespace

import pytest

from restfuldoom_agent.reward import GOAL_PRESETS, Goal, RewardEngine, goal_preset


def state(
    x_fp,
    y_fp,
    health=100,
    kills=0,
    items=0,
    secrets=0,
    enemy_health=None,
    enemy_distance=0,
    combat_target_health=0,
    combat_target_enemy=False,
):
    position = SimpleNamespace(x_fp=x_fp, y_fp=y_fp)
    obj = SimpleNamespace(position=position)
    player = SimpleNamespace(
        object=obj,
        health=health,
        kills=kills,
        items=items,
        secrets=secrets,
    )
    enemies = []
    if enemy_health is not None:
        enemies.append(
            SimpleNamespace(
                object=SimpleNamespace(
                    id=7,
                    health=enemy_health,
                    distance_fp=int(enemy_distance * 65536),
                )
            )
        )
    combat = SimpleNamespace(
        has_shootable_target=combat_target_health > 0,
        target_health=combat_target_health,
        target_is_enemy=combat_target_enemy,
    )
    return SimpleNamespace(player=player, enemies=enemies, combat=combat)


def test_reward_counts_progress_and_combat():
    engine = RewardEngine(Goal(target_x_fp=100, target_y_fp=0))

    reward = engine.score(
        state(0, 0, health=100, kills=0),
        state(50, 0, health=95, kills=1),
    )

    assert reward.kill_delta == 1
    assert reward.health_delta == -5
    assert reward.progress_delta == 50
    assert reward.reward > 0


def test_reward_counts_enemy_damage_before_kill():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, enemy_health=20),
        state(0, 0, enemy_health=5),
    )

    assert reward.kill_delta == 0
    assert reward.damage_delta == 15
    assert reward.reward == 3.0


def test_reward_counts_terminal_enemy_damage_on_kill():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, kills=0, enemy_health=5),
        state(0, 0, kills=1, enemy_health=None),
    )

    assert reward.kill_delta == 1
    assert reward.damage_delta == 5
    assert reward.reward == 11.0


def test_reward_counts_terminal_combat_target_damage_on_kill():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, kills=0, combat_target_health=15, combat_target_enemy=True),
        state(0, 0, kills=1, enemy_health=None),
    )

    assert reward.kill_delta == 1
    assert reward.damage_delta == 15
    assert reward.reward == 13.0


def test_reward_ignores_terminal_non_enemy_combat_target_damage():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, kills=1, combat_target_health=15, combat_target_enemy=False),
        state(0, 0, kills=2, enemy_health=None),
    )

    assert reward.kill_delta == 1
    assert reward.damage_delta == 0
    assert reward.reward == 10.0


def test_reward_ignores_disappeared_enemy_without_kill():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, kills=0, enemy_health=5),
        state(0, 0, kills=0, enemy_health=None),
    )

    assert reward.kill_delta == 0
    assert reward.damage_delta == 0
    assert reward.reward == 0.0


def test_reward_counts_nearest_enemy_distance_progress():
    engine = RewardEngine(goal_preset("combat"))

    reward = engine.score(
        state(0, 0, enemy_health=20, enemy_distance=500),
        state(0, 0, enemy_health=20, enemy_distance=450),
    )

    assert reward.enemy_distance_delta == 50
    assert reward.reward == 0.5


def test_goal_presets_include_demo_modes():
    assert {"survival", "navigation", "combat", "item_collection", "exit_seeking"} <= set(
        GOAL_PRESETS
    )


def test_goal_preset_binds_navigation_target():
    goal = goal_preset("exit-seeking", target_x_fp=512, target_y_fp=-128)

    assert goal.name == "exit_seeking"
    assert goal.target_x_fp == 512
    assert goal.target_y_fp == -128
    assert goal.progress_weight > goal_preset("survival").progress_weight


def test_goal_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown goal preset"):
        goal_preset("speedrun")
