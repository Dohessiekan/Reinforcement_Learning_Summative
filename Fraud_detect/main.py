import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import BankFraudEnv
import pygame
from pygame.locals import *

# Configuration
MODEL_TYPE = "dqn"  # "dqn" or "ppo"
USE_MODEL = True
NUM_EPISODES = 15
RENDER_MODE = "human"

class FraudDetectionRunner:
    def __init__(self):
        self.model_path = f"models/{MODEL_TYPE}dqn_fraud/{MODEL_TYPE}best_model.zip"
        self.algorithm = MODEL_TYPE
        self.render_mode = RENDER_MODE

        # Initialize environment with the same settings as training
        self.base_env = BankFraudEnv(render_mode="rgb_array")
        self.env = DummyVecEnv([lambda: Monitor(self.base_env)])

        if USE_MODEL and os.path.exists(self.model_path):
            print(f"Loading {self.algorithm.upper()} model from: {self.model_path}")
            if self.algorithm == 'dqn':
                self.model = DQN.load(self.model_path, env=self.env)
            else:
                self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("Running with random actions")
            self.model = None

        # Movement control
        self.agent_move_interval = 0.3
        self.fraudster_move_interval = 1.0
        self.last_agent_move = time.time()
        self.last_fraudster_move = time.time()

        # Pygame setup
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption('Bank Fraud Detection Simulation')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)

        # Statistics
        self.total_reward = 0.0
        self.episode_count = 0
        self.steps = 0
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.fraudsters_caught = 0
        self.atm_breaches = 0

    def run_episode(self):
        obs = self.env.reset()
        done = [False]
        episode_reward = 0.0
        exploration_rate = 0.2 if self.model else 1.0

        while not done[0]:
            current_time = time.time()

            if self.render_mode == 'human':
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return False

            # Agent movement
            if current_time - self.last_agent_move >= self.agent_move_interval:
                exploration_rate = max(0.01, exploration_rate * 0.995)

                if self.model and random.random() > exploration_rate:
                    action, _ = self.model.predict(obs, deterministic=False)
                else:
                    action = np.array([random.randint(0, 7)])

                obs, reward, done, info = self.env.step(action)

                if info[0].get('fraud_caught', False):
                    self.fraudsters_caught += 1
                if info[0].get('atm_breached', False):
                    self.atm_breaches += 1

                episode_reward += float(reward[0])
                self.steps += 1
                self.last_agent_move = current_time

                if self.render_mode == 'human':
                    self.render(episode_reward, exploration_rate)
                    self.clock.tick(10)

            # Fraudster movement
            if current_time - self.last_fraudster_move >= self.fraudster_move_interval:
                obs, _, _, _ = self.env.step(np.array([0]))  # NOOP
                self.last_fraudster_move = current_time

            if done[0]:
                self.total_reward += episode_reward
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.cumulative_rewards.append(self.total_reward)
                outcome = info[0].get('outcome', 'unknown')
                print(f"Episode {self.episode_count}: Reward {episode_reward:.1f}, Outcome: {outcome}")
                print(f"  Fraudsters caught: {self.fraudsters_caught}, ATM breaches: {self.atm_breaches}")
                time.sleep(1)
                return True

        return True

    def render(self, current_reward, exploration_rate):
        frame = self.base_env.render()

        if frame is not None:
            if isinstance(frame, np.ndarray):
                frame = np.transpose(frame, (1, 0, 2))
                frame_surface = pygame.surfarray.make_surface(frame)
            else:
                frame_surface = frame

            frame_surface = pygame.transform.scale(frame_surface, (600, 600))
            self.screen.blit(frame_surface, (0, 0))
        else:
            self.screen.fill((255, 255, 255))

        stats = [
            f"Episode: {self.episode_count + 1}",
            f"Step: {self.steps}",
            f"Current Reward: {current_reward:.1f}",
            f"Total Reward: {self.total_reward:.1f}",
            f"Fraudsters Caught: {self.fraudsters_caught}",
            f"ATM Breaches: {self.atm_breaches}",
            f"Algorithm: {self.algorithm.upper()}",
            f"Exploration: {exploration_rate:.1%}" if self.model else "Random Actions"
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()

    def plot_results(self):
        if len(self.episode_rewards) == 0:
            return

        plt.figure(figsize=(12, 6))
        
        # Plot cumulative rewards
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.cumulative_rewards) + 1), self.cumulative_rewards, 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Over Episodes')
        plt.grid(True)
        
        # Plot individual episode rewards
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, 'r-')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Individual Episode Rewards')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Save the plot as JPG
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = f"plots/rewards_{self.algorithm}_{timestamp}.jpg"
        plt.savefig(plot_filename, format='jpg', dpi=300, bbox_inches='tight')
        print(f"\nSaved reward plots to: {plot_filename}")
        
        plt.close()

    def run(self):
        print(f"\n{'='*50}")
        print(f"Starting simulation with {self.algorithm.upper() if USE_MODEL else 'random'} actions")
        print(f"Number of episodes: {NUM_EPISODES}")
        print(f"Reward System:")
        print(f"  +10.0 for catching fraudster (blue on red)")
        print(f"  -20.0 for ATM breach (red on green)")
        print(f"  -0.1 per step")
        print(f"  +5.0 for successful verification")
        print(f"  +15.0 for blocking high-risk fraud")
        print(f"{'='*50}\n")

        for _ in range(NUM_EPISODES):
            if not self.run_episode():
                break

        if self.episode_count > 0:
            print(f"\n{'='*50}")
            print(f"Simulation completed")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total steps: {self.steps}")
            print(f"Total fraudsters caught: {self.fraudsters_caught}")
            print(f"Total ATM breaches: {self.atm_breaches}")
            print(f"Average reward: {np.mean(self.episode_rewards):.1f}")
            print(f"Success rate: {sum(1 for r in self.episode_rewards if r > 0)/self.episode_count:.1%}")
            print(f"{'='*50}")

        if self.render_mode == 'human':
            pygame.quit()

        # Plot and save results after simulation completes
        self.plot_results()

if __name__ == "__main__":
    runner = FraudDetectionRunner()
    runner.run()