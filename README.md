# Learning Airport Surface Scheduling Policies with Masked PPO

COS435 / ECE433 Final Project by Biruk Desta, Hassan Khan, and Keith Torpey.

A centralized PPO agent learns to coordinate aircraft on a synthetic 20-node taxiway
network, evaluated against handcrafted heuristics across five experiments of increasing
complexity: deterministic fixed-route, stochastic variable-count, route-choice
coordination, and bursty Poisson demand. The full pipeline includes the simulator, Gymnasium
environment, baselines, training, evaluation, and visualization.

## Project Explainer Video

For a short visual explanation of the project arc, watch the explainer video:

![Project explainer preview](media/videos/project_explainer/airport_surface_explainer_preview.gif)

The explainer walks through the main story:

1. The original fixed-route MDP was too easy because ConflictAware matched exact
   planning.
2. Route choice introduced a real short-route-vs-bypass tradeoff.
3. Bursty Poisson demand shifted the problem toward backlog/runway metering.
4. MaskablePPO became operationally competitive only after demand-aware
   observations, valid-action masks, strategic hold/no-op, and longer training.

## What This Implements

**Environment:**
- Synthetic 20-node airport graph with 3 contested intersections, 4 gates, runway threshold queue, and timed runway service (3 steps per departure)
- Fixed-route departures and arrivals (Phase 1/2), short-vs-bypass route choice (Exp4), and Poisson continuous demand (Exp5)
- Stochastic arrival times: each aircraft assigned `ready_step ~ Uniform(0, 30)` per episode; aircraft invisible until ready
- Variable-count episodes: N aircraft sampled from a pool at each reset (2–5 depending on scenario)
- Active-slot recycling for Poisson demand: 8 active slots, backlog queue for excess aircraft

**RL:**
- Centralized PPO / MaskablePPO agent: one agent controls all aircraft simultaneously
- Legal-action masks computed per step — agent structurally cannot cause conflicts
- Strategic no-op/hold actions for demand metering under bursty traffic

**Baselines:**
- FCFS, ConflictAware, RunwayAware, route-choice heuristics (AlwaysShort, AlwaysBypass, RouteAware, ArrivalProtected), MPC-H4/H6, legacy ExactPlanner

**Evaluation:**
- 5 experiments, 200 episodes each (50 for Poisson)
- Metrics: reward, completions, conflicts, illegal moves, surface delay, demand delay, throughput, timeout rate, runway utilization

## Experiments

Five experiments of increasing environment complexity. All configured in `experiments.py`.

| ID | Question | Train scenario | Eval scenarios |
|---|---|---|---|
| Exp1 | Does PPO learn and generalize on fixed env? | `default` | dep_only, default, heavy |
| Exp2 | Does masked stochastic training generalize? | `v2_stoch` | v2_det, v2_stoch, v2_heavy |
| Exp3 | Does variable-count training improve OOD? | `v2_variable` | v2_det, v2_stoch, v2_heavy |
| Exp4 | Can PPO learn non-greedy route choice? | `route_choice_trap`, `route_choice_mix` | route_choice_det/trap/mix |
| Exp5 | Does PPO scale to Poisson demand? | `poisson_burst`, `poisson_train_mix` | medium/high/overload/burst |

**Exp1 — Standard PPO on small fixed env (N=200):**

| Policy | Low (2 ac) | Medium (3 ac, training) | High (4 ac) |
|---|---|---|---|
| ConflictAware | +29.2 | +42.0 | **+54.2** |
| PPO (standard, no masking) | +29.2 | +42.0 | **−2338.8** ❌ |

PPO matches on training, times out 100% on one extra aircraft. 178 illegal moves/ep. Standard
PPO memorized a fixed sequence; action masking is required for robust learning.

**Exp2/3 — Masked PPO, stochastic and variable-count (N=200):**

| Policy | v2_det (4 ac) | v2_stoch (4 ac, training) | v2_heavy (5 ac, OOD) |
|---|---|---|---|
| ConflictAware | +54.2 | +56.5 | +66.6 |
| RunwayAware | +54.2 | +56.8 | +67.7 |
| PPO masked, fixed count | +54.2 | **+57.0** | +66.5 |
| PPO masked, variable count (3–5) | +54.2 | +56.9 | **+68.0** |

Masked PPO generalizes to an unseen 5-aircraft scenario with near-zero gap vs ConflictAware.
Variable-count training beats all heuristics on the hardest OOD scenario.

**Exp4 — Route choice, randomized (N=200):**

| Policy | Reward |
|---|---|
| **PPO (route-choice mix) Masked** | **+78.0** |
| RunwayAware | +77.8 |
| MPC-H4 | +77.8 |
| ConflictAware | +76.4 |

PPO trained on diverse route assignments beats all heuristics. PPO trained on a single scenario
fails to generalize (route-choice trap PPO scores +61.2).

**Exp5 — Poisson burst, hardest scenario (N=50):**

| Policy | Censored demand delay | Done/generated | Unserved | Timeout | Surface delay | Reward |
|---|---:|---:|---:|---:|---:|---:|
| ConflictAware | 43.2 | 52.0/53.4 | 1.3 | 32.0% | 8.8 | 175.7 |
| RunwayAware | 43.2 | 52.0/53.4 | 1.3 | 32.0% | 8.8 | 175.4 |
| PPO (burst) Masked | 43.3 | 52.0/53.4 | 1.3 | 32.0% | 8.9 | 173.8 |
| FCFS | 54.5 | 51.7/53.4 | 1.6 | 30.0% | 15.4 | −1134.8 |

Masked PPO matches ConflictAware and RunwayAware on all operational metrics (demand delay,
throughput, unserved, timeout rate). FCFS collapses under burst demand.

Same-seed rollout — RunwayAware vs PPO (demand-aware features + strategic hold):

![Same-seed Poisson burst rollout](experiments/figures/poisson_burst_runwayaware_vs_ppo_hold.gif)

## Main Result

Action masking is necessary for robust PPO on stochastic environments (Exp1 vs Exp2).
Stochastic and variable-count training unlocks out-of-distribution generalization (Exp2/3).
Route-choice diversity training beats all heuristics (Exp4). Masked PPO with demand-aware
observations reaches near-parity with the strongest heuristics under bursty Poisson demand (Exp5).

## Visual Summary

![Final results overview](experiments/figures/final_results_overview_flat.png)

![Poisson burst rollout summary](experiments/figures/poisson_burst_report_rollout_flat.png)

Final visual assets are tracked under `experiments/figures/`:

```text
experiments/figures/final_results_overview_flat.png
experiments/figures/poisson_burst_report_rollout_flat.png
experiments/figures/poisson_burst_runwayaware_vs_ppo_hold.gif
```

The rendered project explainer video is under `media/videos/project_explainer/`.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv rl
source rl/bin/activate
pip install -r requirements.txt
```

## Quick Checks

Verify the simulator:

```bash
python simulator.py
```

Verify the Gymnasium environment:

```bash
python -c "from env.airport_env import AirportEnv; from stable_baselines3.common.env_checker import check_env; check_env(AirportEnv()); print('env OK')"
```

## Training and Evaluation

Train all models (skips existing checkpoints automatically):

```bash
python training/train.py
```

Train specific models only:

```bash
python training/train.py --only ppo_v2_stoch ppo_route_choice_mix
python training/train.py --retrain   # force retrain everything
```

Run all 5 experiments and save CSVs:

```bash
python evaluation/evaluate.py
```

Generate per-experiment figures:

```bash
python visualization/plots.py
```

Generate the same-seed rollout animation (requires seed1 checkpoint):

```bash
python visualization/rollout_animation.py
```

Checkpoints → `experiments/models/`  
Results (CSV) → `experiments/results/`  
Figures → `experiments/figures/`

## Repository Structure

```text
simulator.py              Core simulator and scenario logic
experiments.py            Experiment/model/scenario registry
baselines/                Heuristic, MPC, and exact-planner baselines
env/                      Gymnasium wrapper and action masks
training/                 PPO/MaskablePPO training entry point and configs
evaluation/               Evaluation runner and metric aggregation
visualization/            Plot and rollout-animation scripts
experiments/figures/      Selected final visual assets
media/videos/             Optional rendered explainer videos
```

## Code Attribution

RL training uses off-the-shelf PPO from Stable-Baselines3 and MaskablePPO from
SB3-Contrib. The simulator, Gymnasium wrapper, legal-action masks, route-choice
and Poisson scenarios, handcrafted baselines, metrics, plots, and evaluation
scripts are project-specific implementations for this class project.
