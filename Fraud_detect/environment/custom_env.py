import numpy as np
import gymnasium as gym
import pygame
import random
from typing import Optional


class BankFraudEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size: int = 5, render_mode: Optional[str] = None):
        super().__init__()
        self.size = grid_size
        self.render_mode = render_mode
        self.atm_pos = np.array([2, 2])  # Center position

        # Flattened observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(12,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(
            8)  # 4 movements + 4 detection actions

        # Initialize state
        self.agent_pos = None
        self.fraudsters = None
        self.current_step = 0
        self.max_steps = 100

        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 100
            self.screen = pygame.display.set_mode(
                (self.size*self.cell_size, self.size*self.cell_size))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Place agent randomly (not on ATM)
        while True:
            self.agent_pos = np.array([
                random.randint(0, self.size-1),
                random.randint(0, self.size-1)
            ])
            if not np.array_equal(self.agent_pos, self.atm_pos):
                break

        # Initialize 2-4 fraudsters at edges
        self.fraudsters = []
        edges = ['top', 'bottom', 'left', 'right']
        for _ in range(random.randint(2, 4)):
            edge = random.choice(edges)
            if edge == 'top':
                pos = (random.randint(0, self.size-1), 0)
            elif edge == 'bottom':
                pos = (random.randint(0, self.size-1), self.size-1)
            elif edge == 'left':
                pos = (0, random.randint(0, self.size-1))
            else:  # right
                pos = (self.size-1, random.randint(0, self.size-1))
            self.fraudsters.append(np.array(pos))

        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten all positions into a single vector
        obs = np.zeros(12, dtype=np.float32)
        obs[0:2] = self.agent_pos
        obs[2:4] = self.atm_pos

        # Add fraudster positions (pad with -1 if less than 3 fraudsters)
        for i in range(3):
            if i < len(self.fraudsters):
                obs[4+i*2:6+i*2] = self.fraudsters[i]
            else:
                obs[4+i*2:6+i*2] = [-1, -1]

        # Add normalized step count
        obs[10] = self.current_step / self.max_steps

        # Add threat level (0-3 based on closest fraudster)
        min_dist = min(np.linalg.norm(self.atm_pos - f)
                       for f in self.fraudsters) if self.fraudsters else self.size*2
        obs[11] = 3 if min_dist <= 1 else 2 if min_dist <= 2 else 1 if min_dist <= 3 else 0

        return obs

    def step(self, action):
        reward = -0.05  # Small penalty per step
        terminated = False
        info = {}
        self.current_step += 1

        # Movement actions (0-3)
        if action < 4:
            direction = [
                np.array([0, -1]),  # Up
                np.array([1, 0]),   # Right
                np.array([0, 1]),   # Down
                np.array([-1, 0])   # Left
            ][action]
            new_pos = self.agent_pos + direction
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.agent_pos = new_pos

                # Reward for moving toward nearest fraudster
                if self.fraudsters:
                    nearest = min(
                        self.fraudsters, key=lambda f: np.linalg.norm(self.agent_pos - f))
                    old_dist = np.linalg.norm(self.agent_pos - nearest)
                    new_dist = np.linalg.norm(new_pos - nearest)
                    reward += 0.2 if new_dist < old_dist else -0.1

        # Detection actions (4-7)
        else:
            detection_type = action - 4
            if detection_type == 0:  # Basic monitoring
                reward -= 0.5
            elif detection_type == 1:  # Verify transaction
                for i, fraudster in enumerate(self.fraudsters):
                    dist = np.linalg.norm(self.agent_pos - fraudster)
                    if dist <= 1.5:
                        reward += 10 if dist <= 1.0 else 5
                        self.fraudsters.pop(i)
                        info['fraud_caught'] = True
                        break
                else:
                    reward -= 2
            elif detection_type == 2:  # Block transaction
                if any(np.linalg.norm(f - self.atm_pos) <= 1.5 for f in self.fraudsters):
                    reward += 15
                    self.fraudsters = [
                        f for f in self.fraudsters if np.linalg.norm(f - self.atm_pos) > 1.5]
                else:
                    reward -= 5
            elif detection_type == 3:  # Full investigation
                reward -= 3
                if len(self.fraudsters) > 0:
                    reward += 8
                    self.fraudsters = []

        # Move fraudsters toward ATM
        for i in range(len(self.fraudsters)):
            if random.random() < 0.7:  # 70% chance to move toward ATM
                direction = np.sign(self.atm_pos - self.fraudsters[i])
            else:  # 30% random move
                direction = np.array(
                    [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])])

            new_pos = self.fraudsters[i] + direction
            new_pos = np.clip(new_pos, 0, self.size-1)
            self.fraudsters[i] = new_pos

            # Check if fraudster reached ATM
            if np.array_equal(new_pos, self.atm_pos):
                reward -= 20
                terminated = True
                info['atm_breached'] = True

        # Check termination conditions
        if len(self.fraudsters) == 0:
            terminated = True
            info['all_fraudsters_stopped'] = True

        if self.current_step >= self.max_steps:
            terminated = True
            info['timeout'] = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return

        if not hasattr(self, "screen"):
            pygame.init()
            self.cell_size = 100
            self.screen = pygame.display.set_mode((self.size*self.cell_size,
                                                   self.size*self.cell_size))
            pygame.display.set_caption("Bank Fraud Detection")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (x * self.cell_size, 0),
                             (x * self.cell_size, self.size * self.cell_size))
        for y in range(self.size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (0, y * self.cell_size),
                             (self.size * self.cell_size, y * self.cell_size))

        # Draw ATM (green square at center)
        atm_rect = pygame.Rect(
            self.atm_pos[0] * self.cell_size + 2,
            self.atm_pos[1] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, (0, 200, 0), atm_rect)

        # Draw agent (blue circle)
        agent_center = (
            int(self.agent_pos[0] * self.cell_size + self.cell_size // 2),
            int(self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        )
        pygame.draw.circle(self.screen, (0, 0, 255),
                           agent_center, self.cell_size // 3)

        # Draw fraudsters (red circles)
        for fraudster in self.fraudsters:
            fraudster_center = (
                int(fraudster[0] * self.cell_size + self.cell_size // 2),
                int(fraudster[1] * self.cell_size + self.cell_size // 2)
            )
            pygame.draw.circle(self.screen, (255, 0, 0),
                               fraudster_center, self.cell_size // 3)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2))

    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
