# Reinforcement_Learning_Summative



![env gif](rotation.gif)







Bank Fraud Detection RL Environment - Overview
This project provides a reinforcement learning environment for simulating bank fraud detection scenarios, where an agent patrols a grid to intercept fraudsters attempting to breach ATMs. The implementation includes:

A custom Gymnasium environment with grid-based mechanics

Fraudster movement patterns and ATM breach dynamics

Multiple action types (movement, monitoring, interventions)

3D visualization using PyOpenGL

Training scripts for DQN and Policy Gradient methods

Model saving/loading functionality

Setup Instructions
To run this project locally:

Clone the repository:
git clone [repository-url]
cd project_root
Install dependencies:

pip install -r requirements.txt
Run the main script:

python main.py
Key dependencies include Gymnasium, Stable-Baselines3, PyOpenGL, and Pygame. The environment supports both training new models and visualizing pre-trained agents. Configuration options are available in the main script for adjusting grid size, fraudster behavior, and training
