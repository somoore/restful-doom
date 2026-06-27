from types import SimpleNamespace

import pytest

from restfuldoom_agent.reward import GOAL_PRESETS, Goal, RewardEngine, goal_preset


def state(x_fp, y_fp, health=100, kills=0, items=0, secrets=0):
    position = SimpleNamespace(x_fp=x_fp, y_fp=y_fp)
    obj = SimpleNamespace(position=position)
    player = SimpleNamespace(
        object=obj,
        health=health,
        kills=kills,
        items=items,
        secrets=secrets,
    )
    return SimpleNamespace(player=player)


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
