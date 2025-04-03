import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import BankFraudEnv

# Setup directories
os.makedirs("models/dqn_fraud", exist_ok=True)
os.makedirs("logs/dqn_fraud", exist_ok=True)

# Create environment
env = DummyVecEnv([lambda: Monitor(BankFraudEnv(render_mode=None))])

# Optimized hyperparameters for MlpPolicy
hyperparams = {
    "learning_rate": 2.5e-4,
    "buffer_size": 100000,
    "learning_starts": 10000,
    "batch_size": 128,
    "tau": 1.0,
    "gamma": 0.97,
    "target_update_interval": 1000,
    "train_freq": (4, "step"),
    "gradient_steps": 1,
    "exploration_fraction": 0.25,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {
        "net_arch": [256, 128],
        "activation_fn": torch.nn.ReLU,
        "optimizer_kwargs": {"weight_decay": 1e-4}
    }
}

model = DQN(
    "MlpPolicy",  # Using MlpPolicy with flattened observations
    env,
    verbose=1,
    tensorboard_log="logs/dqn_fraud",
    **hyperparams
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="models/dqn_fraud",
    name_prefix="dqn_fraud",
    save_replay_buffer=True
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/dqn_fraud",
    log_path="logs/dqn_fraud",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

# Training
model.learn(
    total_timesteps=300000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
    tb_log_name="dqn_fraud_run",
    log_interval=4
)

# Save final model
model.save("models/dqn_fraud/dqn_fraud_final")
print("Training completed. Model saved to models/dqn_fraud/")
env.close()
