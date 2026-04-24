# Learning Conflict-Aware Control Policies for Airport Surface Operations

COS435 — Hassan Khan, Keith Torpey, Biruk Desta

A reinforcement learning agent for airport surface control. A centralized PPO agent jointly decides gate releases and intersection movement in a small synthetic taxiway network, learning to coordinate aircraft while minimizing delays and avoiding conflicts.

---

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv rl
source rl/bin/activate
pip install -r requirements.txt
```

---

## Run Order

**1. Verify the simulator works**
```bash
python simulator.py
```

**2. Verify the Gym environment**
```bash
python -c "from env.airport_env import AirportEnv; from stable_baselines3.common.env_checker import check_env; check_env(AirportEnv()); print('env OK')"
```

**3. Train PPO**
```bash
python training/train.py
```

**4. Run evaluation**
```bash
python evaluation/evaluate.py
```

**5. Generate plots**
```bash
python visualization/plots.py
```

---

## File Structure

```
simulator.py          # core MDP logic (no Gym dependency)
env/airport_env.py    # Gymnasium wrapper
baselines/            # FCFS and conflict-aware heuristic policies
training/             # PPO training + hyperparameter configs
evaluation/           # metrics and evaluation runner
experiments/          # saved models, TensorBoard logs, results
```
