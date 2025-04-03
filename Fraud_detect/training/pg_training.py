import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import BankFraudEnv

# Setup directories
os.makedirs("models/ppo_fraud", exist_ok=True)
os.makedirs("logs/ppo_fraud", exist_ok=True)

# Create environment
env = DummyVecEnv([lambda: Monitor(BankFraudEnv(render_mode=None))])

# Corrected callback for training monitoring
class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Safe way to get current observations
        if 'obs' in self.locals:
            obs = self.locals['obs']
            if hasattr(self.model.policy, "get_distribution"):
                obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                entropy = dist.entropy().mean().item()
                self.logger.record("train/entropy", entropy)
        return True

# Optimized hyperparameters
hyperparams = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
        "optimizer_kwargs": {
            "weight_decay": 1e-4,
        }
    }
}

# Initialize model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/ppo_fraud",
    **hyperparams
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="models/ppo_fraud",
    name_prefix="ppo_fraud",
    save_replay_buffer=False
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/ppo_fraud",
    log_path="logs/ppo_fraud",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

monitor_callback = TrainingMonitorCallback()

# Training configuration
training_kwargs = {
    "total_timesteps": 300000,
    "callback": [checkpoint_callback, eval_callback, monitor_callback],
    "progress_bar": False,  # Disabled to avoid shutdown errors
    "tb_log_name": "ppo_fraud_run",
    "log_interval": 4
}

# Training
try:
    model.learn(**training_kwargs)
finally:
    # Ensure proper cleanup
    model.save("models/ppo_fraud/ppo_fraud_final")
    print("Training completed. Model saved to models/ppo_fraud/")
    env.close()