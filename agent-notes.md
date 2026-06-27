# Agent Brain Notes

## Milestones

- Started structured local brain implementation: fast protobuf-state policy loop, persistent map/enemy memory, reward-scored episode summaries, and simple parameter evolution.
- Added Codex MCP surface for `brain_drive` and `brain_memory` so future runs can be controlled through the local tool bridge instead of screenshots.
- Added Docker-backed MCP trainer design: restart Doom for each candidate, run a fresh protobuf rollout, score with rewards, and promote policy parameters only when they beat stored memory.
- First live Docker training batch completed: three fresh candidates ran and memory was written, but the policy got stuck and overused `ACTION_USE`; no kills or pickups yet.
- Updated the policy with a multi-phase stuck recovery sequence and fitness credit for explored cells and visible enemy ticks so evolution has useful signal before first kill.
- Second live Docker training batch showed improvement: candidates picked up an item, visited more cells, had far fewer stuck events, and promoted better parameters. Still no kills.
- Found a memory bug: enemy `last_seen_tick` from prior fresh episodes could look newer than the current episode tick, causing stale enemy chasing. Fixed by ignoring remembered enemies with future ticks.

## Current Hypothesis

The first useful intelligence should be a deterministic skill-selection policy backed by memory, not an LLM driving individual tics. The policy should learn enough from real rollouts to avoid repeating obvious failures, then later become a training target for a small learned model.

## Good State Definition

Training reached the first "good" state on E1M1: autonomously complete one level traversal and score at least one kill in that same run. The current verified run is `brain-5b63907febc4` from `brain-train-68-current-success.jsonl`: level transition to E1M2, 6 peak kills, 3 peak items, and 60 end health. The next target is repeatability and learned skill selection beyond behavior cloning.

## Export / Resume Contract

- Training job bundle schema: `restfuldoom.training_job.v1`.
- Bundle contents: manifest, `agent_memory/e1m1.json`, `agent-notes.md`, learned skill model checkpoints, and referenced trajectory JSONL files.
- Resume path: import the bundle into a cloud worker or Hellbox/Shrink job, then continue structured-brain episodes against the new gRPC endpoint using the promoted parameters from memory.

## Latest Milestones

- Third live Docker training batch improved further: best fitness reached `-11.42`, best candidate picked up 2 items, visited 29 cells, and reduced stuck events to 61. Still no kills and no level completion, so the training target is not met.
- Added explicit success criteria to the brain: success requires at least one kill and level completion by default.
- Added portable export/import for training jobs so progress can move from Docker to cloud.
- Added protobuf-driven `hunt_known_enemy`: the brain now uses all streamed living enemy coordinates, not screenshots or REST door lists, to close distance toward combat. Fitness now includes nearest-enemy distance progress.
- Fourth live Docker training batch showed known-enemy hunting reduced distance by about 435 units but got pinned against geometry with no line of sight. Added blocked-target memory so the brain stops chasing a known enemy through the same wall from the same cell.
- Added engine-side protobuf navigation probes instead of using screenshots or REST door lists: `forward_open`, side/back openness, `use_line_ahead`, and front blocking distance/special. The brain now uses those probes to use doors, sidestep, or turn when the direct route is blocked.
- Added engine-side combat probes and ray-fan navigation. Early batches confirmed the agent could see and approach enemies from structured state, but it still stayed outside pistol range until short-term contact-corridor memory kept it moving through the opening after line of sight dropped.
- First real combat breakthrough: live Docker batch `brain-train-15-directional-use` candidate 3 dealt 40 enemy damage, scored 3 kills, reached 237 visible-enemy ticks, and promoted run `brain-86b41d87a2f7` with score `386.6738`. Not a good state yet because the level was not completed.
- Next hypothesis: use nearby special/use-line data from protobuf to make door/switch/exit interactions explicit enough for the promoted combat policy to finish E1M1 rather than only clearing enemies.
- Longer promoted-policy run `brain-train-16-use-lines-long` candidate 1 improved again: 45 enemy damage, 3 kills, 8 items, 80 visited cells, max position about `x=2480, y=-2128`, and promoted run `brain-eea454042350` with score `417.3823`. Still no level completion by 8,000 states.
- Found that using special-line midpoint distance causes bad door/switch behavior on long linedefs. Added nearest-point-on-line data to protobuf/FFI so the brain can use actual segment distance and angle.
- Added bounded `seek_known_enemy`, manual-use special filtering, line-attempt blacklisting, and route-to-progression movement over the protobuf direction-probe fan. This moved the agent from wandering into reliable first-contact combat and post-kill progression attempts.
- Best run so far: `brain-train-25-exit-assist` / `brain-329df98b66bf` promoted with score `419.3735`: 6 kills, 2 items, 95 visited cells, alive at max state, and reached the exit approach around `y=-4641`. Not a good state yet because the map did not transition.
- Regression found: applying front-side line targeting to all manual specials made the policy worse and caused an early death. Narrowed side-aware behavior to local exit switches only; ordinary doors now use the earlier successful behavior.
- New best rollout `brain-train-31-critical-retreat` / `brain-c560bed0a254` promoted with score `436.1415`: 6 kills, 2 items, alive with 60 health at 16,000 states, but still no level transition. The final blocker was pressing line 330 from about 168 map units away.
- Added learned hazard-cell handling after repeated deaths on damaging floor cells around line 195. The policy now records non-combat health loss in memory and has explicit hazard escape skills. First hazard escape prevented the immediate floor death, but over-broad hazard routing distorted the path; narrowed it toward progression lines and close-walk-trigger skipping.
- Added critical-health defensive fire after a late death against enemy id 9 showed the agent retreating passively while aligned at close range. The policy now shoots while backing/strafe at critical health.
- Current exit-door evidence: line 330 is visible and selected in cell `23:-36`, but cannot be activated from ~168 units. Line 325 is a close front-side manual door and should be aligned/used before returning to line 330. Added tests for close angled assist-door turn/use and stuck manual-line use around door lines 340/341.
- Strongest failed rollouts now reliably reach the final exit room: candidates 58, 60, 61, and 62 cleared the map enemies with 6 kills but stalled at line 330. The current blocker is retrying close assist doors 324/325 after line-attempt blacklisting; added an exit-specific retry path that can press line 325 with a small forward nudge once the exit push stalls.
- Candidate 65 exposed a separate mid-map regression: after 2 kills the policy chased far walk trigger line 308 from ~1900 units away and stopped hunting the remaining monsters. Tightened progression readiness so distant walk triggers wait until more enemies are cleared, while known-enemy seeking remains active until roughly the 5-kill route.
- Candidate 66 achieved the first good state: autonomous E1M1 completion with 6 kills on the route, assist-door retries, and a final line 330 activation into E1M2. Candidate 67 then exposed a close-exit ordering bug where line 330 at distance 16/angle 0 still routed instead of pressing; moved `press_exit_switch` ahead of blocked-front routing and fixed stats to preserve peak kills/items across map-reset counters.
- Candidate 68 verified the current code after that fix: `success=true`, `level_completed=true`, `kill_delta=6`, `peak_kills=6`, `item_delta=3`, `peak_items=3`, and `press_exit_switch=2`. Trained the first learned skill selector from this trajectory: `agent_models/skill-policy.json`, schema `restfuldoom.skill_policy.v1`, 2,961 samples, 29 classes, train accuracy `0.8982`, eval accuracy `0.882`.

## Open Risks

- Repeatability still needs improvement; recent current-code runs can complete E1M1, but route timing varies and final door/exit sequencing remains the highest-risk segment.
- Pistol firing cadence may need raw ticcmd or carefully timed shoot pulses if high-level `ACTION_SHOOT` is too lossy.
- The learned skill selector is behavior cloning, not independent RL yet. It predicts skill choices from successful protobuf trajectories and is exported for cloud resume, but the deterministic controller still owns live tic-level actions.
