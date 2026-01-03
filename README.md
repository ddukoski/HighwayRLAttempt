# HighwayRLAttempt

This repository is a small, practical collection of training scripts and helpers for experimenting with `highway-env` and `stable-baselines3`. The goal is to make it easy to run quick experiments, save the best runs, and export clear plots for analysis.

## What you'll find here

The project has a few simple pieces that work together: environment helpers under `environments/`, agent factories in `agents/`, the main training runner `training.py`, a focused `parking.py` script that uses HER+SAC for goal-conditioned parking tasks, and a lightweight plotting utility in `scripts/` that reads TensorBoard event files and exports PNG/PDF graphics.

## Getting started (fast)

Create and activate a Python 3 virtual environment, then install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To kick off the default highway training run, use the Makefile target:

```bash
make train-highway
```

If you want to run the parking experiment (HER+SAC):

```bash
make train-parking
```

## How it works

`training.py` creates a training and an evaluation environment with `make_env()` from `environments/environment_utils.py`, builds an agent from the `agents` factory, and uses an `EvalCallback` to save the best model during training. Logging is written to `./logs` so the plotting tools can find TensorBoard event files.

To select an environment, simply modify `ENV_NAME` in `training.py`:

```python
ENV_NAME = EnvRegistry.<desired_environemnt>
```

Where `<desired_environemnt>` is any environment belonging to `EnvRegistry` in `/environments/environment_utils`. Currently, the model being selected is only tuned for `highway`, `highway_fast` and `parking`, you may modify this as needed in `training.py`.

### Loading pretrained models

If you already have a saved model (for example `parking-v0.zip` or `highway-fast-v0.zip`) place it at the repository root and the runner will load it automatically. Example code for programmatic loading:

```python
from environments.environment_utils import make_env, EnvRegistry
from agents.agents import BaseHighwayAgents

env = make_env(EnvRegistry.parking, config={})
agent = BaseHighwayAgents.make_her_sac_agent(env)
agent = type(agent).load('parking-v0.zip', env=env)

# for highway (DQN)
env = make_env(EnvRegistry.highway_fast, config={})
agent = BaseHighwayAgents.make_dqn_agent(env)
agent = type(agent).load('highway-fast-v0.zip', env=env)
```

### Exporting plots

The plotting tool in `scripts/tensorboard_plot.py` will read event files under `./logs` and save PNG (and PDF via the publication helper) into `./logs/plots`. A quick way to export current runs is:

```bash
make plot
```

Currently, for the pretrained models, the plots already exist in `/saved_results`.