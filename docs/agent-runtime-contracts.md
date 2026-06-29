# Agent Runtime Contracts

This document answers four implementation questions that matter for training
and cloud resume. Every contract below maps to code, JSONL fields, or exported
schemas.

For the fuller operator-facing interface spec, including exact data ownership,
clock boundaries, skill option semantics, and the protobuf-to-observation gap
register, see `docs/agent-training-interface.md`.

## 1. Decision Layer To Fast Controller

The learned decision layer does not drive raw Doom input. It chooses one
high-level skill index per macro-step, and the fast controller turns that skill
into one protobuf `PlayerAction`.

Runtime trace:

| Phase | Code | Data | Writes |
| --- | --- | --- | --- |
| Observe | `SkillController.observation(state)` | `restfuldoom.observation.v1` vector | none |
| Mask | `DoomAgentEnv.action_mask()` -> `SkillController.action_mask(state)` | `restfuldoom.skill_action_mask.v1` booleans | none |
| Decide | `PPOTrainer` / `ActorCritic` | integer index into `restfuldoom.skill_action.v1` | rollout `action`, `action_mask`, logprob |
| Execute | `SkillController.action_for(index, state)` | latest protobuf `GameState`, selected skill | one protobuf `PlayerAction` and `info.decision` |
| Simulate | `DoomAgentEnv.step()` | action stream over `DoomAgentClient` | next `GameState`, reward, done, metadata |
| Remember | `SkillController.record_action_history()` | macro-step outcome | next observation history features |

The boundary is strict:

- PPO sees feature vectors, masks, rewards, and trajectory metadata.
- `SkillController` sees a selected skill index and the current protobuf state.
- `BrainPolicy` owns the local mechanics: aim, movement, firing cadence, use
  timing, stuck recovery, and fallback behavior.
- PPO never sends raw `ticcmd` and never receives controller internals such as
  logits or gradients.

The exported schema for this handshake is
`restfuldoom_agent.schemas.DECISION_CYCLE_SCHEMA`
(`restfuldoom.decision_cycle.v1`). Rollout rows include a compact
`info.decision_cycle` payload with input/output ticks and schema ids, so a
training artifact can be audited without replaying the code.

Rollout rows also include `info.learning_trace` using
`restfuldoom.learning_trace.v1`. It names the important pre-action observation
groups, feasible skills, selected skill, controller primitive, and reward
outcome so contact/route/combat failures can be inspected without manually
mapping vector indexes back to protobuf-derived features.

## 2. Memory Layer

`AgentMemory` is not a hidden neural state. It is a JSON world ledger and
training checkpoint index, persisted at `agent_memory/e1m1.json` using schema
`restfuldoom.agent_memory.v1`.

Concrete persisted shape:

```json
{
  "schema": "restfuldoom.agent_memory.v1",
  "cells": {
    "23:-36": {
      "visits": 42,
      "enemy_sightings": 3,
      "damage_events": 0,
      "last_seen_tick": 12345
    }
  },
  "enemies": {
    "57": {
      "last_seen_tick": 12345,
      "last_position": [1526.1, -2538.7],
      "last_distance": 717.0,
      "last_health": 20,
      "line_of_sight": true
    }
  },
  "episodes": [],
  "policy": {},
  "learned_policy": {},
  "ppo_policy": {},
  "ppo_best_checkpoint": {},
  "ppo_checkpoints": []
}
```

Concrete query paths:

| Query | Reader | Purpose |
| --- | --- | --- |
| `AgentMemory.best_params()` | deterministic trainer | load promoted controller parameters |
| `AgentMemory.remembered_enemies(x, y, tick, max_age_tics)` | `extract_features()` | recover recent enemy sightings when line of sight drops |
| `AgentMemory.summary()` | CLI / MCP operator surface | inspect progress and failures |

Concrete update paths:

| Update | Writer | Purpose |
| --- | --- | --- |
| `AgentMemory.record_step(...)` | deterministic `run_brain_episode()` | write cells, enemy sightings, damage events, and lessons |
| `AgentMemory.finish_episode(...)` | deterministic `run_brain_episode()` | append compact episode summary and promote params |
| `ppo_agent._record_ppo_checkpoint(...)` | PPO training | write latest checkpoint, best resume candidate, checkpoint lineage, and rollout summary |
| `ppo_agent._record_eval_history(...)` | PPO eval | write promotion-gate outcomes |
| `train_skill_policy_from_memory(...)` | behavior cloning | write learned selector metadata |

PPO does not write persistent memory on every macro-step. During collection,
memory-backed features are read-only; episode-local controller history is
updated in `SkillController`. Persistent writes happen after checkpoints,
evals, or deterministic episode completion. This keeps the inner loop fast and
makes resume/export auditable.

## 3. Skill Definitions

Skills are currently code-defined options, not learned functions and not
external config. The stable action space is:

| Index | Skill | Learned Today | Code-Owned Today |
| --- | --- | --- | --- |
| 0 | `engage` | whether to approach/strafe a visible enemy | aim and movement primitive |
| 1 | `fire` | when a shot opportunity is worth taking | target validation, cooldown, attack pulse |
| 2 | `seek_enemy` | when remembered pursuit is useful | enemy selection and pursuit primitive |
| 3 | `open_use_line` | when to act on a door/switch/use affordance | line targeting, turn/use timing |
| 4 | `route_progression` | when route exploration beats combat | waypoint/progression movement |
| 5 | `retreat` | when danger warrants distance | backing/strafe mechanics |
| 6 | `recover_stuck` | when to interrupt for recovery | unstuck sequence |
| 7 | `press_exit` | when exit affordances should dominate | exit alignment and use |
| 8 | `close_visible_contact` | when visible contact needs closure | contact-ray movement and recent corridor follow-through |

The machine-readable definition is
`restfuldoom_agent.schemas.ACTION_SCHEMA` (`restfuldoom.skill_action.v1`).
Each checkpoint and rollout buffer carries the action schema, including index,
name, controller entrypoint, role, primary signal, fallback, mask semantics,
and evolution rule. Checkpoint resume treats existing action indexes as a trust
boundary: appended actions can be migrated, but the saved action prefix must
match the current schema exactly or the checkpoint is rejected.

The important rule: PPO learns the selector, not the primitive. A future skill
may become config-backed or internally learned only if the same index/name
contract remains valid for old checkpoints.

Two current mask rules are intentionally controller-heavy. If the protobuf
combat probe reports a shootable enemy and the controller can fire, the mask
follows through to `fire` and suppresses normal engage/use/route actions. If
an enemy is visible but not shootable, the mask exposes `close_visible_contact`
instead of generic `engage`; `open_use_line` is only available when the contact
line is close/use-ready or the enemy contact is close and aligned. If
PPO selects `open_use_line` for a recent visible-contact manual line, the mask
keeps that option active until the line is no longer valid or a shootable enemy
appears. That contact-line follow-through is bounded: after a long same-skill
streak the mask releases back to other contact actions unless the line is close
enough to press. PPO still has to create those opportunities; the controller
protects the short option windows once they exist without letting them become
unbounded loops.

## 4. Protobuf State To Learning Observation

The protobuf stream is richer than screenshots but still not equal to a full
learning observation. The current observation contract is
`restfuldoom.observation.v1`; it combines:

- live protobuf state: player, level, enemies, combat probe, navigation probes,
  use lines, current sector, route waypoint
- memory queries: remembered enemies and blocked-target state
- controller state: stuck status and previous macro-action summary
- bounded temporal context: movement delta, enemy-distance trend,
  route-distance trend, cell streak, recent visible/shootable flags, route
  progress/failure
- contact context: recent visible-contact activity plus current or remembered
  contact use-line distance, angle, close-use readiness, follow-through state,
  and age

This is enough for local combat-start PPO: from-scratch masked PPO learned to
kill from a validated combat reset. It is not enough yet for full spawn-to-exit
competence. Current live evidence says spawn-only PPO can make route progress,
but still often fails to bridge from spawn to first shootable contact.

Observation gap register:

| Gap | Current Signal | Missing Signal | Next Implementation |
| --- | --- | --- | --- |
| Spawn to first combat | route waypoint, temporal route progress, remembered enemies, local projected-cell topology context | compact route graph or progressed-map snapshot | add topology graph features or Hellbox/Shrink snapshot curriculum |
| Contact to shootable | visible enemy distance, contact use-line memory, first-shootable reward, explicit contact-line observation, local topology context | doorway/contact local geometry over time | validate shootable reset stage or add contact-ray/topology features |
| Combat target quality | shootable yes/no, target distance, enemy health | aim error, weapon range quality, cooldown window | extend protobuf combat probe |
| Survival threat | sector hazard, health deltas | projectile and incoming-damage prediction | add projectile/threat affordances |
| Replayability | `seed_applied=true` reset evidence and fresh starts | progressed map state | verify fixed/held-out seed gates and snapshot restore |

The promotion rule should stay strict while this is incomplete. Reward shaping
can train intermediate curricula, but a promoted full policy still needs level
completion, kills, survival, and baseline comparison.
