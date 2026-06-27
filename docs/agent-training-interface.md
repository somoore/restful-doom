# Agent Training Interface Spec

This is the current implementation contract for learning agents. It answers
what talks to what, what is persisted, what a skill is, and what is still
missing between the protobuf stream and a rich learning observation.

## 1. Decision Layer And Fast Controller

The learned policy does not drive Doom at the raw tic level. It chooses one
stable skill index for a bounded macro-step. The fast controller converts that
skill into one protobuf `PlayerAction`, usually with a short `duration_tics`.

The clocks are separate:

| Clock | Owner | Rate | Responsibility |
| --- | --- | --- | --- |
| Doom tic | engine | 35 Hz | simulate, publish protobuf `GameState`, apply queued `PlayerAction` |
| macro-step | `DoomAgentEnv` | 1 to `max_action_tics` Doom tics | score one learned decision |
| PPO update | `PPOTrainer` | after rollout collection | update selector weights from rewards |

The concrete call order is:

1. `DoomAgentEnv.action_mask()` calls `SkillController.action_mask(state)`.
2. PPO receives `restfuldoom.observation.v1` plus the boolean mask.
3. PPO samples or selects one integer action index.
4. `DoomAgentEnv.step(action_index)` calls
   `SkillController.action_for(action_index, state)`.
5. `SkillController` calls the matching `BrainPolicy` primitive and returns one
   protobuf `PlayerAction`.
6. `DoomAgentEnv` sends that action over `DoomAgentClient.GameSession`.
7. The environment consumes the resulting protobuf states for the action
   duration, aggregates reward, route/contact metadata, and terminal state.
8. `SkillController.record_action_history()` stores episode-local context for
   the next observation.

Only four data types cross the decision/controller boundary:

| Direction | Payload | Schema or Code |
| --- | --- | --- |
| environment to actor | observation vector | `restfuldoom.observation.v1` |
| environment to actor | feasible-action mask | `restfuldoom.skill_action_mask.v1` |
| actor to controller | integer skill index | `restfuldoom.skill_action.v1` |
| controller to Doom | one protobuf action | `PlayerAction` |

The controller never receives PPO gradients, logits, or optimizer state. PPO
never sends raw `ticcmd` values directly. The mask used during sampling is
stored in the rollout buffer and reused during PPO logprob recomputation.

Every live transition carries `info.decision_cycle`:

```json
{
  "schema": "restfuldoom.decision_cycle.v1",
  "observation_schema": "restfuldoom.observation.v1",
  "action_schema": "restfuldoom.skill_action.v1",
  "memory_contract": "restfuldoom.agent_memory_contract.v1",
  "input_tick": 1234,
  "output_tick": 1238,
  "macro_tics": 4
}
```

That makes one JSONL row enough to audit which observation, mask, selected
skill, controller decision, and Doom tick range produced a reward.

Collected PPO rows also carry `info.learning_trace`, schema
`restfuldoom.learning_trace.v1`. The trace is intentionally compact: it groups
the important observation features by player/combat/navigation/use-line/route/
memory/temporal/survival concerns, lists the feasible skills from the mask, and
records the selected skill, executed controller decision, reward, route/contact
outcomes, and transition deltas.

Rollout summaries aggregate the trace contact group into:

- `contact_context_active_steps`
- `contact_use_line_active_steps`
- `contact_use_line_close_steps`
- `contact_use_line_followthrough_steps`
- `mean_contact_use_line_distance_norm`
- `mean_contact_use_line_age_norm`

## 2. Memory Layer

`AgentMemory` is an inspectable JSON ledger, not a neural hidden state. The
default path is `agent_memory/e1m1.json`, schema
`restfuldoom.agent_memory.v1`.

The persisted document is shaped like this:

```json
{
  "schema": "restfuldoom.agent_memory.v1",
  "cells": {
    "23:-36": {
      "first_seen_tick": 100,
      "visits": 42,
      "enemy_sightings": 3,
      "damage_events": 0,
      "last_seen_tick": 12345,
      "last_seen_at": "2026-06-27T12:00:00Z"
    }
  },
  "enemies": {
    "57": {
      "first_seen_tick": 100,
      "last_seen_tick": 12345,
      "last_position": [1526.1, -2538.7],
      "last_distance": 717.0,
      "last_health": 20,
      "line_of_sight": true,
      "visible_count": 8,
      "max_threat": 4.2
    }
  },
  "episodes": [],
  "policy": {},
  "learned_policy": {},
  "ppo_policy": {},
  "ppo_best_checkpoint": {},
  "ppo_checkpoints": [],
  "lessons": []
}
```

Query paths used inside the loop:

| Query | Reader | Use |
| --- | --- | --- |
| `AgentMemory.best_params()` | deterministic trainer | load promoted controller parameters |
| `AgentMemory.remembered_enemies(...)` | `extract_features()` | recover recent enemy sightings after line of sight drops |
| `AgentMemory.summary()` | CLI/MCP operator | inspect progress without opening the full JSON |

Update paths:

| Update | Writer | Timing |
| --- | --- | --- |
| `AgentMemory.record_step(...)` | deterministic brain | every deterministic rollout step |
| `AgentMemory.finish_episode(...)` | deterministic brain | after deterministic episode completion |
| `ppo_agent._record_ppo_checkpoint(...)` | PPO trainer | after a checkpointed PPO update |
| `ppo_agent._record_eval_history(...)` | PPO evaluator | after promotion-gate eval |
| `train_skill_policy_from_memory(...)` | behavior cloning | after supervised selector training |

PPO collection does not write persistent memory on every macro-step. It reads
memory-derived features, mutates only `SkillController` episode-local history,
then writes memory at checkpoint or evaluation boundaries. That keeps the inner
loop fast and keeps resume/export state auditable.

When `--checkpoint-eval-curriculum` is enabled, checkpoint-boundary writes also
include `checkpoint_eval` with schema
`restfuldoom.ppo_checkpoint_curriculum_eval.v1`. `ppo_best_checkpoint` records
`checkpoint_selection_source` so resume/export consumers can distinguish a
rollout-only choice from a cross-stage curriculum-evaluated choice.

Export and cloud resume copy:

- `agent_memory/e1m1.json`
- referenced PPO checkpoints
- trajectory or rollout buffers
- schema metadata embedded in the checkpoint/buffer

## 3. Skill Definitions

A skill is currently a code-defined option with a stable exported descriptor.
It is not a free-form function selected by an LLM, not external config yet, and
not a learned movement primitive yet.

The stable action space is:

| Index | Skill | Learned Today | Code-Owned Today |
| --- | --- | --- | --- |
| 0 | `engage` | whether to approach visible contact | turn, move, strafe, continue contact corridor |
| 1 | `fire` | when to spend a shot window | target validation, cooldown, attack pulse |
| 2 | `seek_enemy` | when remembered pursuit is useful | target selection and pursuit primitive |
| 3 | `open_use_line` | when to use a door/switch affordance | line targeting, turn/use timing |
| 4 | `route_progression` | when route movement beats other options | progression-line movement and exploration |
| 5 | `retreat` | when danger warrants distance | backward/strafe mechanics |
| 6 | `recover_stuck` | when recovery should interrupt | unstuck sequence |
| 7 | `press_exit` | when exit affordances dominate | exit alignment and use |

The machine-readable definition is `ACTION_SCHEMA`
(`restfuldoom.skill_action.v1`). Each descriptor includes the index, name,
kind, learned flag, execution owner, controller entrypoint, primary signal, and
fallback. Checkpoints and rollout buffers carry this schema.

PPO learns selector weights:

- action probability for each skill
- value estimate for the observation
- advantage/reward signal for choosing the skill

The controller owns primitive mechanics:

- movement amount and turn amount
- aiming tolerance and firing cadence
- door/switch use timing
- short option follow-through for shot windows and recent contact use-lines
  with a same-skill streak guard unless the remembered line is close enough to
  activate
- stuck recovery
- route/contact fallbacks

A future skill can become config-backed or internally learned only if the
stable index/name option contract remains valid for older checkpoints.

## 4. Protobuf State To Rich Observation

The protobuf stream is the source of truth, but it is not automatically a rich
learning observation. The current pipeline is:

1. `GameState` publishes player, enemy, level, combat, navigation, sector, and
   route-waypoint affordances.
2. `extract_features()` normalizes those fields into `TacticalFeatures` and
   queries `AgentMemory.remembered_enemies(...)`.
3. `features_from_tactical()` converts tactical state into base numeric
   features.
4. `SkillController` appends macro-action history.
5. `SkillController` appends bounded temporal context.
6. `SkillController` appends explicit recent-contact/use-line context.
7. `SkillController` appends local topology/cell-visit context.
8. `SkillController` appends explicit visible-contact target geometry.
9. PPO consumes the final 104-feature `restfuldoom.observation.v1` vector.

The current observation covers:

- player health, ammo, kills, items, position, and facing
- live visible, known, and remembered enemy signals
- combat probe target validity and distance
- front/back/left/right navigation probes
- nearby use-line and exit-line affordances
- current sector damage/hazard fields
- topology frontier count from open direction probes leading to low-visit
  nearby cells
- one route waypoint with distance, angle, priority, and type
- previous skill, shootable opportunity, and route outcome history
- short temporal deltas for movement, enemy distance, route distance, contact,
  and route failure ratio
- current or remembered contact use-line state: active flag, distance, angle,
  close-use readiness, bounded follow-through flag, and age
- local topology context: current-cell visits, least-visited open projected
  neighbor direction, and open-neighbor exhaustion ratio
- visible-contact geometry: active/shootable/needs-closure flags, nearest
  visible enemy distance, angle, alignment, and close-contact state

The remaining learning gap is real. Known missing pieces:

| Gap | Why It Matters | Current Workaround | Needed Upgrade |
| --- | --- | --- | --- |
| topology graph | spawn-to-contact requires memory of route structure | one local waypoint, coarse cells, frontier count, and local projected-cell context | compact graph observation |
| progressed-map state | fresh teleport starts do not replay opened doors/enemy movement | named fresh-reset curriculum | save-state or Hellbox/Shrink snapshot restore |
| recurrent context | feed-forward PPO sees only hand-built short history | bounded temporal features | recurrent policy if features stay insufficient |
| combat target quality | shootable yes/no hides aim margin and weapon quality | fire mask and action reward | richer combat probe fields |
| projectile threat | survival lacks incoming-threat prediction | health/sector deltas | projectile and incoming-damage affordances |
| deterministic replay | reset seed is currently a label | run metadata records seed request | verified `seed_applied=true` |

The promotion rule stays strict while those gaps are open. Dense rewards can
train intermediate checkpoints, but promotion still requires level completion,
kills, survival, and baseline comparison.

Within-rollout curriculum mixing is available as a short-run regularizer:
`--rollout-stage-mix round_robin|random` changes reset starts between
`done=True` episodes inside one PPO buffer, and
`--rollout-stage-segment-tics` can shorten each reset episode so a small buffer
contains multiple stages. Rollout rows keep per-record `curriculum_stage`
metadata, and summaries report `curriculum_stage_counts`. This helps a single
PPO update see several fresh-reset stages, but it does not solve the
progressed-map gap; snapshot restore is still required for true mid-trajectory
curriculum.

Snapshot-backed curriculum is represented by
`restfuldoom.snapshot_curriculum.v1` manifests. `ppo_agent
--snapshot-curriculum <path> --snapshot-restore-command <argv-template>` loads
those stages into the regular `restfuldoom.ppo_curriculum.v1` schedule. Snapshot
stages set `reset_mode: snapshot`; `DoomAgentEnv.reset()` skips
`ResetEpisode`, reconnects the gRPC stream, and waits for the first live state
after the external restore command completes. Each transition carries
`reset_context` with schema `restfuldoom.reset_context.v1`, including
`source=snapshot_restore`, the episode index, snapshot ID/path/digest,
restore timing/exit metadata, expected state, and the actual first observed
state. Training exports include `snapshot_curriculum`,
`snapshot_restore_context`, and any local snapshot artifact files referenced by
the manifest.

The manifest can be generated from existing JSONL trajectory evidence before
the VM artifacts exist:

```bash
PYTHONPATH=agent python -m restfuldoom_agent.snapshot_builder \
    --trajectory trajectories/brain-train-68-current-success.jsonl \
    --output trajectories/e1m1-snapshot-curriculum.json \
    --name e1m1-progressed-bottlenecks \
    --snapshot-dir snapshots \
    --save-slot-base 3 \
    --auto first-visible \
    --auto first-enemy-shootable \
    --auto first-damage
```

That output is a capture plan until every referenced artifact is present and
validated:

```bash
PYTHONPATH=agent python -m restfuldoom_agent.snapshot_curriculum \
    trajectories/e1m1-snapshot-curriculum.json \
    --validate \
    --require-artifacts
```

`--save-slot-base` assigns native Doom agent save slots to generated stages.
Stages with `snapshot.slot` or `snapshot.ref: "save_slot:N"` restore through
the gRPC `LoadSnapshot` RPC and do not require an external restore command.
Stages without a slot still use `--snapshot-restore-command` for
Hellbox/Shrink or file-backed artifacts.

Native save-slot manifests can also be captured directly from a live
structured-brain rollout:

```bash
PYTHONPATH=agent python -m restfuldoom_agent.snapshot_capture \
    --token-json trajectories/agent-doom-token.json \
    --tls \
    --memory-path agent_memory/e1m1.json \
    --trajectory-jsonl trajectories/snapshot-capture.jsonl \
    --output trajectories/e1m1-snapshot-curriculum.json \
    --name e1m1-progressed-bottlenecks \
    --save-slot-base 3 \
    --attempts 3 \
    --reset-before-attempt \
    --reset-seed-base 31 \
    --auto first-visible \
    --auto first-enemy-shootable \
    --auto first-damage \
    --auto first-kill \
    --verify-loads
```

The capture command uses the deterministic structured brain only to reach
progressed states. It saves native `agentdoomN.dsg` slots at the first matching
protobuf milestones and emits a normal `restfuldoom.snapshot_curriculum.v1`
manifest. PPO still learns independently from reward feedback when the manifest
is later passed to `ppo_agent --snapshot-curriculum`. Use `--attempts` with
`--reset-before-attempt` when one route rollout is not reliable enough; each
attempt uses a fresh `BrainPolicy` instance and, when reset is enabled, queues a
fast gRPC `ResetEpisode` before streaming. Multi-attempt captures write
per-attempt trajectory files and record `source.attempt_reports` plus
per-stage `capture_attempt` metadata. Native save slots are limited to `0..9`,
so `--save-slot-base + requested_milestones - 1` must stay inside that range.
Use `first-enemy-shootable` for combat curriculum; plain `first-shootable`
matches any shootable target and can capture non-enemy targets when the protobuf
combat probe reports `target_is_enemy=false`.

Snapshot resets are verified by default. After an external restore or native
`LoadSnapshot`, `DoomAgentEnv.reset()` observes the first protobuf state and
compares it with the stage `expected_state`. The verifier compares only fields
present in the manifest, with tolerances for saved `level_time` and position
drift, and writes `restfuldoom.snapshot_restored_state_verification.v1` into
`reset_context.restored_state_verification`. PPO training fails the reset when
verification is invalid unless `--no-snapshot-verify-restored-state` is set for
debugging. `GameState.tick` is a stream counter, not a saved level timestamp,
so it is advisory by default; `--snapshot-verify-stream-tick` enables it only
for diagnostics.

Native slots can be managed with `python -m restfuldoom_agent.snapshot_slots
save|load --slot N` or the Hellbox wrapper commands `snapshot-save` and
`snapshot-load`. The Hellbox wrapper `snapshot-capture` runs the live capture
flow using the current token JSON and `SNAPSHOT_AUTO` selectors.

Snapshot PPO summaries separate inherited map counters from earned progress.
Absolute `max_kills` can be nonzero immediately after loading a progressed
`first-kill` slot, so summaries also report `kill_delta`, `max_kill_gain`,
`snapshot_kill_delta`, and `snapshot_max_kill_gain`. Rollout-only best
checkpoint selection scores use the earned fields. This keeps a checkpoint from
looking better merely because a reset slot already contained kills.

Checkpoint curriculum eval follows the same contract. Each eval stage is reset
through `_env_config_for_stage()`, so snapshot stages restore native save slots
or external artifacts instead of falling back to teleport-like fresh starts.
Serialized `EpisodeEval` rows include `reset_source`, start/end episode/map,
`start_kills`, absolute `max_kills`, earned `kill_delta`/`max_kill_gain`, and
item/secret deltas/gains. The aggregate `mean_kills` used for eval selection is
earned after reset; inherited snapshot kills remain visible only as start-state
metadata.

The validator checks the schema, stage structure, local snapshot files, and
`sha256:` digests. PPO experiments can load an unvalidated manifest for plumbing
smokes, but any promotion or export meant to resume in cloud should use a
validated manifest so the training bundle carries real progressed-state
artifacts rather than teleport approximations. Native save-slot curricula should
record the slot refs in the manifest and verify the first restored protobuf
state through `reset_context.actual_first_state`.
