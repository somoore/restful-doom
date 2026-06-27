# Agent Learning Architecture

This repo has two agent layers that should stay separate:

1. The fast controller turns one selected skill into safe tic-level Doom input.
2. The decision layer chooses which skill should run next.

The current training path intentionally learns the second layer first. Directly
learning raw `ticcmd` at 35 Hz is possible later, but it makes early PPO spend
most of its time rediscovering aiming, door use, and stuck recovery. Skill-level
learning gives the model a smaller action space while still using the real
protobuf stream and real Doom rewards.

## Runtime Stack

```mermaid
flowchart TD
    State["gRPC GameState stream"] --> Features["Feature extraction"]
    Features --> Decision["Decision layer"]
    Decision --> Skill["Selected skill"]
    Skill --> Controller["Fast controller"]
    Controller --> Action["PlayerAction / raw ticcmd"]
    Action --> Doom["Doom simulation"]
    Doom --> State

    State --> Memory["AgentMemory"]
    Decision --> Memory
    Memory --> Features
    Memory --> Decision
    Controller --> History["Macro-action history"]
    History --> Features
```

The loop has these concrete implementations:

- `DoomAgentClient` owns the bidirectional gRPC stream.
- `BrainPolicy` is the hand-built fast policy plus controller.
- `SkillController` wraps `BrainPolicy` for PPO by exposing eight stable skill
  actions.
- `DoomAgentEnv` is the Gym-style environment used by PPO.
- `SkillPolicyModel` is the behavior-cloned softmax selector trained from good
  trajectory decisions.
- `PPOTrainer` is the independent RL learner for skill selection.

## Fast Controller vs Decision Layer

The fast controller is the code that knows how to execute local movement and
combat safely. It handles:

- aiming toward a visible or remembered enemy
- firing only when the combat probe says the shot is valid
- retreating and strafing around close enemies
- using doors and switches
- avoiding blocked geometry
- escaping stuck cells and damaging floor cells

The decision layer is the code that chooses the next intent. Today there are
three decision sources:

- `BrainPolicy`: deterministic heuristic decision source used for successful
  baseline runs.
- `SkillPolicyModel`: behavior-cloned classifier that predicts the skill the
  heuristic would have selected from a successful trajectory.
- `PPOTrainer`: actor-critic model that independently chooses one of the PPO
  skill actions from reward feedback.

The bridge is an explicit macro-step handshake:

1. `DoomAgentEnv.step(action_index)` receives one PPO skill action.
2. `DoomAgentEnv.action_mask()` derives the feasible option set from the
   current protobuf state, memory, and transient controller state.
3. The PPO actor receives the observation plus mask and returns an action
   index. PPO also stores that same mask in the rollout buffer, then reuses it
   during the update when recomputing logprobs.
4. `SkillController.action_for(action_index, state)` extracts tactical features
   from the latest protobuf `GameState` and returns one concrete protobuf
   `PlayerAction`.
5. `DoomAgentEnv` sends that action over the bidirectional gRPC stream and
   waits through its bounded `duration_tics`.
6. The environment aggregates reward, transition metadata, and terminal state
   across those tics.
7. `SkillController.record_action_history()` stores the previous skill,
   shootable-target opportunity, and same-skill streak for the next
   observation.
8. The next PPO observation is current protobuf state plus memory-derived
   features plus that macro-action history.

That contract is exported as `restfuldoom.decision_cycle.v1` through
`restfuldoom_agent.schemas.DECISION_CYCLE_SCHEMA`. Rollout buffers and
checkpoints now carry the full contract. Each live `EnvStep.info` also includes
a compact `decision_cycle` object with the schema id, observation schema,
action schema, memory contract id, input tick, output tick, and macro tic
count. A single JSONL trajectory row therefore tells us exactly how the
decision layer, controller, memory, and Doom stream interacted.

The concrete runtime payload is:

| Step | Producer | Consumer | Payload |
| --- | --- | --- | --- |
| observe | `DoomAgentEnv` | PPO actor | `restfuldoom.observation.v1` feature vector and `restfuldoom.skill_action_mask.v1` boolean mask |
| decide | PPO actor | `DoomAgentEnv` | integer action index into `restfuldoom.skill_action.v1` |
| execute | `SkillController` | `DoomAgentClient` | one protobuf `PlayerAction` plus `rollout_record.info.decision` metadata |
| score | `DoomAgentEnv` | PPO buffer and memory | reward, transition deltas, route outcome, shootable-target flags, and schema markers |
| remember | `SkillController` / PPO trainer | next observation and `AgentMemory` | episode-local macro history plus persistent checkpoint/eval summaries |

The fast controller and decision layer therefore communicate only through
stable skill indexes, masks, protobuf state, and JSONL metadata. The controller
does not receive gradients or policy logits, and PPO does not send raw ticcmds.

The division of responsibility is intentionally strict:

| Boundary | Owns | Does not own |
| --- | --- | --- |
| PPO actor | choosing a skill index under the mask | raw turning, door timing, firing cadence |
| `SkillController` | converting a skill index into a safe local `PlayerAction` | long-term promotion or checkpointing |
| `BrainPolicy` | tactical primitives and heuristics used by the controller | gradient updates |
| `DoomAgentEnv` | reset, macro-step execution, reward aggregation, trajectory metadata | deciding which skill is good |
| `AgentMemory` | persistent world/training ledger and explicit query/update APIs | neural hidden state |

`SkillController.reset_episode_context()` clears the action-history features at
episode boundaries and after heuristic reset warmup, so PPO does not inherit
stale controller state from a previous episode.

`SkillController.action_mask(state)` is the feasibility contract for PPO
sampling. It derives a boolean mask from protobuf combat/navigation affordances
and memory:

- `fire` is available only when the combat probe reports a shootable enemy and
  the controller cooldown allows a shot.
- `engage` is available when an enemy is visible.
- `retreat` is available only for low health or close threats.
- `seek_enemy` is available when no enemy is visible but memory/protobuf has a
  target to hunt.
- `open_use_line`, `route_progression`, `recover_stuck`, and `press_exit` are
  enabled only when their corresponding affordances are present.

The PPO actor samples under this mask and the PPO update recomputes logprobs
under the same mask from the rollout buffer. This keeps exploration independent
while avoiding impossible or irrelevant skills.

## Skill Representation

The PPO skill action space is in `restfuldoom_agent.schemas.PPO_SKILL_ACTIONS`:

- `engage`
- `fire`
- `seek_enemy`
- `open_use_line`
- `route_progression`
- `retreat`
- `recover_stuck`
- `press_exit`

These are currently code-defined options, not learned movement primitives. Each
skill is implemented as a branch in `SkillController._execute_skill()` and
delegates to small controller methods in `BrainPolicy`.

The action space is now also exported as data through
`restfuldoom_agent.schemas.ACTION_SCHEMA` using schema
`restfuldoom.skill_action.v1`. Checkpoints, rollout buffers, and training-job
bundles carry a definition for every skill: index, name, controller entrypoint,
role, primary signal, fallback, representation, execution owner, and mask
semantics. This is enough for a future cloud worker or MCP surface to inspect
"what action 3 means" without importing private controller internals.

The current representation is deliberately conservative:

- Skills are code-defined options in `SkillController._execute_skill()`.
- `ACTION_SCHEMA` is the exported definition contract for checkpoints,
  rollout buffers, and training bundles.
- PPO learns selection only: probability of choosing a skill and the value of
  that choice under reward.
- The controller owns the primitive mechanics: local aim, movement, use-line
  timing, fire cadence, and fallback behavior.

This means `fire` is not a learned function yet. It is a stable option whose
availability is masked by combat affordances, while PPO learns when selecting
that option is valuable. Later, a skill can become config-backed or dispatch to
a learned subpolicy, but only if it preserves the same index/name contract for
old checkpoints.

The learned policies do not learn the movement primitive itself yet. They learn
when to call the primitive. That keeps the first independent RL objective
tractable and lets existing protobuf affordances carry most low-level Doom
knowledge.

The current representation is:

| Skill | Representation | Learned part | Code-owned part |
| --- | --- | --- | --- |
| `engage` | code-defined option | when to approach/strafe a visible enemy | aiming and movement primitive |
| `fire` | code-defined option | when a shot opportunity is worth taking | shoot cooldown, target validation, raw button press |
| `seek_enemy` | code-defined option | when remembered enemy pursuit is useful | enemy-memory selection and route primitive |
| `open_use_line` | code-defined option | when a use affordance should be acted on | line targeting, turn/use timing |
| `route_progression` | code-defined option | when exploration/progression beats combat | line/frontier/probe routing |
| `retreat` | code-defined option | when danger warrants distance | backward/strafe retreat mechanics |
| `recover_stuck` | code-defined option | when recovery should interrupt other plans | unstuck turn/backtrack sequence |
| `press_exit` | code-defined option | when the exit should take priority | exit switch alignment and use |

Later, a skill can become learned internally without changing the PPO action
index if it preserves the same option contract. For example, `engage` could
eventually dispatch to a learned local-navigation subpolicy while the top-level
PPO still chooses action index `0`.

## Memory Contract

`AgentMemory` persists to `agent_memory/e1m1.json` using schema
`restfuldoom.agent_memory.v1`. It is a JSON document that can be exported with a
training job and resumed in Docker, Hellbox, or a cloud worker.

The important sections are:

- `cells`: visit counts, enemy sightings, damage events, and last-seen ticks for
  coarse map cells.
- `enemies`: last seen position, health, distance, line of sight, and threat per
  enemy id.
- `episodes`: compact summaries from recent real rollouts.
- `policy`: best deterministic parameters, best score, success flags, and the
  promoted run id.
- `learned_policy`: behavior-cloned skill selector metadata.
- `ppo_policy`: latest PPO checkpoint metadata, reward config, rollout summary,
  and eval history.
- `ppo_checkpoints`: exported PPO checkpoint lineage.

The concrete contract is exported as
`restfuldoom_agent.schemas.MEMORY_CONTRACT` with schema
`restfuldoom.agent_memory_contract.v1`. Rollout buffers and PPO checkpoints
carry it next to the observation/action schemas.

Memory has named update and query paths:

| Path | Type | Called by | Purpose |
| --- | --- | --- | --- |
| `AgentMemory.best_params()` | query | structured-brain training | load the promoted deterministic controller parameters |
| `AgentMemory.remembered_enemies(x, y, tick, max_age_tics)` | query | `extract_features()` | return recent enemy sightings by id, position, last tick, and current distance while rejecting stale or future-tick records |
| `AgentMemory.summary()` | query | `brain_agent --memory-summary`, MCP `brain_memory` | return compact diagnostics for operator inspection |
| `AgentMemory.record_step(features, decision, reward, stats)` | update | `run_brain_episode()` | update cells, enemy sightings, damage events, and per-episode stats after each real transition |
| `AgentMemory.finish_episode(stats, params, promoted)` | update | `run_brain_episode()` | append rollout summary, promote deterministic params when warranted, and store lessons |
| `ppo_agent._record_ppo_checkpoint()` | update | PPO training | store the latest checkpoint, reward config, rollout summary, and checkpoint lineage |
| `ppo_agent._record_eval_history()` | update | PPO evaluation | append candidate/baseline promotion-gate outcomes |
| `train_skill_policy_from_memory()` | update | behavior-cloning bootstrap | write learned skill-selector metadata |

Memory is not a neural hidden state. It is an explicit, inspectable world and
training ledger. The learned model gets compact features derived from current
protobuf state plus selected memory-derived features such as remembered enemies,
stuck state, and blocked targets.

The memory lifecycle is:

1. On reset, PPO and deterministic training read checkpoint/policy metadata to
   resume the requested policy.
2. Before feature extraction, `extract_features()` queries remembered enemies
   and blocked-target state, then combines that with the latest protobuf
   `GameState`.
3. During controller execution, only episode-local history is mutated; the
   persistent JSON memory is not written on every PPO macro-step.
4. After training or evaluation, the trainer writes checkpoint lineage, rollout
   summaries, and promotion-gate outcomes to memory.
5. Export bundles copy the memory JSON plus checkpoint files so Docker,
   Hellbox, or cloud workers can resume from the same learning state.

## Observation Contract

PPO receives the feature vector declared in
`restfuldoom_agent.schemas.OBSERVATION_SCHEMA`. It is derived from protobuf
state, memory, and macro-action history, not screenshots. The current schema has
69 features: 54 base tactical features plus 15 action-history features.

The base feature groups are:

- player health, ammo, kills, and items
- normalized map position and facing
- visible, known, and remembered enemy counts
- nearest enemy distance, angle, threat, and health
- combat probe target validity and distance
- local navigation probes for front/back/side openness
- usable-line and exit-line affordances
- stuck and blocked-target indicators
- current sector damage/hazard affordances
- route waypoint distance, angle, priority, and type

The action-history group is:

- one-hot previous PPO skill
- whether the previous macro-step had a shootable target
- same-skill streak normalized by 8
- whether the previous macro-step selected `route_progression`
- distance gained or lost toward the previous route waypoint
- whether the previous route waypoint was reached or failed
- consecutive failed route-attempt count

The schema now also declares source groups:

- `protobuf_state`: live player, enemy, combat, navigation, use-line, and level
  fields from `GameState`, including current-sector hazard and route-waypoint
  probes.
- `memory_queries`: remembered enemies and blocked-target state.
- `controller_state`: stuck detection and macro-action history.

This is good enough for early skill learning, but it is not yet a complete
learning observation. Known gaps:

- No compact topological map graph, only local probes plus coarse cell memory.
- Only one macro-step of action history plus a route-failure streak; there is
  no recurrent state or longer temporal context yet.
- Route waypoints are a single local progression target, not a full route plan
  or topology graph.
- No enemy projectile or incoming-damage prediction.
- No deterministic RNG seed application yet; reset seed is currently a label,
  not replay proof.

These gaps explain why the first PPO runs show distance-progress reward before
damage or kills. The model can learn "approach enemy" from the current vector,
but it needs stronger combat-opportunity features and action-aware rewards to
learn "fire now" reliably.

The most important current gap is spawn-to-first-combat. Recent spawn-only PPO
buffers show positive route progress and route reward, but still zero
shootable-target steps, zero damage, and zero kills. That means protobuf state
is rich enough for local movement reward, but not yet rich enough to make the
fresh-spawn route reliably produce the first valid combat affordance. The next
two credible fixes are true progressed-state snapshot restore or a compact
topology/temporal observation that remembers route attempts across more than
one macro-step.

The next observation upgrades should be staged rather than speculative:

1. Add a short temporal window or recurrent policy after that is stable;
   one previous macro action is not enough to represent "I just tried this
   door" or "this corridor approach failed twice."
2. Add a compact topology graph once true save-state or Hellbox/Shrink snapshot
   restore is available, because map graph learning is much more useful when
   we can repeatedly resume from progressed map states.

## Promotion Rule

Reward shaping is not enough for promotion. A PPO checkpoint is only promotable
when it beats the baseline gate and satisfies the project success floor:

- completion rate at or above the configured minimum, default `1.0`
- mean kills at or above the configured minimum, default `1.0`
- no regression on survival, time-to-exit, or stuck events
- reward is compared, but reward alone cannot promote a checkpoint

This keeps dense shaping useful for training without confusing movement reward
with actual game competence.

## Behavior Cloning Bootstrap

PPO can be warm-started from successful protobuf trajectories before live RL
updates. The bootstrap path:

1. Reads trajectory JSONL records with `metadata.policy_decision.skill`.
2. Maps rich expert skills into the eight stable PPO skills using
   `EXPERT_TO_PPO_SKILL_ACTION`.
3. Trains the actor with supervised cross-entropy.
4. Continues normal PPO collection and updates against live Doom reward.

This is not the final intelligence layer. It is a curriculum tool that gets the
policy closer to useful behavior than a uniformly random skill selector. Live
reward and the promotion gate still decide whether the checkpoint is useful.

## Checkpoint Resume

`ppo_agent --resume-checkpoint <path>` loads a saved
`restfuldoom.ppo_checkpoint.v1` checkpoint and continues live PPO collection
with the model weights, optimizer state, observation schema, action schema, and
update index from that file. The CLI validates that checkpoint observation and
action dimensions match the current exported schemas before collecting more
rollouts.

## Reset Curriculum

`DoomAgentEnv` has two curriculum mechanisms.

First, it can ask the server to apply a fresh-reset `EpisodeStart` through
`ResetEpisode`. The start can include:

- player position
- explicit angle or `face_nearest_enemy`
- health and armor
- starting ammo

The C game loop applies this on the simulation thread after `G_DeferedInitNew`
and before publishing the next protobuf state. This is not a save-state restore:
the level is still freshly reset, so opened doors, enemy movement, and map
mutations from a later trajectory are not replayed. It is useful for cheap
combat starts and should eventually be replaced or complemented by true
snapshot restore for long-route curriculum.

The PPO CLI also has named reset-start schedules. The first one,
`e1m1-spawn-to-combat`, uses `restfuldoom.ppo_curriculum.v1` metadata and
rotates fresh-reset starts from easiest to hardest:

1. `combat_start`: known legal combat start with immediate enemy line-of-sight.
2. `combat_wide_left`: validated fresh-reset combat variant with immediate
   shootable target.
3. `combat_back_left`: validated fresh-reset combat variant with a deeper
   approach angle and immediate shootable target.
4. `fresh_spawn`: true E1M1 spawn with no teleport override.

Earlier trajectory-derived route starts were removed from the named curriculum
because live validation showed they were not valid fresh-reset states. A
coordinate taken from a successful trajectory does not replay opened doors,
enemy movement, or map mutations. Those starts need save-state or
Hellbox/Shrink snapshot restore before they are valid PPO curriculum stages.

Modes are `round_robin`, `progressive`, and `random`. Rollout records,
checkpoint extras, and memory checkpoint entries carry both the curriculum
descriptor and the active stage. This makes spawn-to-contact transfer auditable:
we can see whether an update trained combat, first contact, route approach, or
fresh spawn.

Second, the environment can optionally run heuristic-only warmup after reset and
before PPO starts collecting transitions. Warmup can stop on:

- first visible enemy
- first shootable combat target
- step limit
- hard tic limit
- death or level change

The rollout summary reports warmup steps, tics, and stop reasons. Current local
evidence shows naive warmup from the default E1M1 spawn is too expensive for the
inner PPO loop and often hits the tic limit before reaching combat.

The useful fresh-reset combat start found so far is:

- Doom units: `x=3248`, `y=-3280`
- fixed-point: `x_fp=212860928`, `y_fp=-214958080`
- flags: `face_nearest_enemy=true`, `health=100`, `ammo_bullets=50`

From that start, protobuf reports line of sight to three enemies and an
immediate shootable target after reset.

## Current Learning Evidence

The deterministic structured brain has already met the project's first good
state: complete E1M1 and kill enemies along the way. PPO is still in transition.

Recent PPO evidence:

- `ppo-episode-start-combat-smoke`: BC-warm PPO from the fresh combat start got
  `max_kills=1` in all three 128-transition updates, with `damage_delta` 60,
  65, and 70. This proves the PPO environment/reward/controller loop can score
  combat, but it is not pure independent learning because BC initialized the
  actor.
- `ppo-independent-combat-start-smoke`: PPO from scratch improved fire
  selections from 15/128 to 43/128, action reward from `1.91` to `11.73`, total
  reward from `12.4953` to `23.2953`, and late-update damage back to 40. It did
  not score a kill in eight short updates. This is the first independent
  reward-driven learning signal, not a promotable policy.
- `ppo-tight-mask-independent-combat-start-smoke`: PPO from scratch with action
  masks reached `max_kills>=1` in all eight 128-transition updates and
  `max_kills=2` in update 7. Deterministic eval of checkpoint
  `ppo-tight-mask-independent-combat-start-smoke-ppo-0007.pt` repeated one kill
  in 3/3 combat-start eval episodes and beat masked-random on mean reward
  (`70.8597` vs `41.7365`) with survival 1.0. This is a promoted
  combat-curriculum result, not full-level competence.
- `ppo-transfer-spawn-smoke`: resumed that combat checkpoint from normal E1M1
  spawn for two 256-transition updates. It reduced nearest-enemy distance by
  about 650 units with `route_progression` and `seek_enemy`, but still produced
  `shootable_target_steps=0`, `damage_delta=0`, and `max_kills=0`. The next
  gap is spawn-to-first-contact curriculum.
- `ppo-spawn-trend`: resumed the combat checkpoint from normal E1M1 spawn for
  six 128-transition updates. Route attempts made positive cumulative waypoint
  progress in every buffer and route rewards stayed positive, but the run still
  had zero shootable-target steps, zero damage, and zero kills. This confirms
  the current bottleneck is route-to-contact observation/curriculum, not the
  combat-start PPO controller.

## Next Architecture Work

The next useful changes are:

- Extend action history beyond one macro-step or add a recurrent policy.
- Add true save-state or Hellbox/Shrink snapshot restore so PPO can train from
  progressed map states, not only fresh-reset teleport starts.
- Promote combat affordances from binary target presence to richer target
  quality, including aim error, weapon range, and cooldown.
- Transfer the independently trained combat checkpoint into a spawn-to-exit
  curriculum and evaluate against the deterministic full-level baseline.
- Move skill definitions from exported descriptors to optional external config
  after the action set stabilizes.
- Wire true deterministic seed application in the Doom reset path.
