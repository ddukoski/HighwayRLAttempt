import os
import multiprocessing
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from agents.agents import BaseHighwayAgents
from environments.environment_utils import EnvRegistry, make_env
from scripts.tensorboard_plot import watch_and_plot, collect_all_runs, plot_runs, make_publication_plot

ENV_NAME = EnvRegistry.parking
NUM_EPISODES_EVAL = 10
LOG_DIR = "./logs"

if ENV_NAME == EnvRegistry.parking:
    TRAIN_STEPS = int(5e4)
elif ENV_NAME in (EnvRegistry.highway, EnvRegistry.highway_fast):
    TRAIN_STEPS = int(5e4)
else:
    TRAIN_STEPS = int(8e4)

env_config = {}

if ENV_NAME == EnvRegistry.highway_fast:
    env_config = {
        "vehicles_count": 30,
        "duration": 40,
        "collision_reward": -1,
        "high_speed_reward": 0.4,
    }

elif ENV_NAME.startswith("intersection"):
    env_config = {
        "vehicles_count": 15,
        "duration": 30,
        "collision_reward": -5,
    }

elif ENV_NAME == EnvRegistry.parking:
    env_config = {
        "vehicles_count": 1,
        "duration": 40,
    }

train_env = make_env(ENV_NAME, render_mode=None, config=env_config)
eval_env = make_env(ENV_NAME, render_mode=None, config=env_config)

if ENV_NAME == EnvRegistry.parking:
    agent = BaseHighwayAgents.make_her_sac_agent(
        train_env,
        tensorboard_log=LOG_DIR,
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=1024,
        tau=0.05,
        buffer_size=int(1e6),
        learning_starts=500,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
    )
else:
    agent = BaseHighwayAgents.make_dqn_agent(
        train_env,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
    )

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=3000,
    deterministic=True,
    render=False,
)

model_path = f"{ENV_NAME}.zip"

plot_proc = None
if os.path.exists(model_path):
    agent = type(agent).load(model_path, env=train_env)
else:
    plots_dir = os.path.join(LOG_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    

    agent.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)
    agent.save(model_path)

    

    from scripts import keep_best_run

    best, best_max = keep_best_run.select_best_run(LOG_DIR)
    archive = keep_best_run.archive_other_runs(LOG_DIR, best)

    runs = collect_all_runs(LOG_DIR)
    plot_runs(runs, plots_dir)
    if best:
        pub_out = make_publication_plot(best, tag="rollout/ep_rew_mean", outpath=None)
mean_reward, std_reward = evaluate_policy(
    agent,
    eval_env,
    n_eval_episodes=20,
    deterministic=True,
)
 
render_env = make_env(ENV_NAME, render_mode="human", config=env_config)

for episode in range(NUM_EPISODES_EVAL):
    obs, info = render_env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated

render_env.close()
