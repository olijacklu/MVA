from gymnasium.wrappers import TimeLimit
from fast_env_py import FastHIVPatient
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
import time


env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=1000
)

class ProjectAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        
        # Initialize XGBoost models for each action
        self.models = [None for _ in range(self.action_dim)]
        self.scalers = [StandardScaler() for _ in range(self.action_dim)]
        
        # Enhanced XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'lambda': 1.5,
            'alpha': 0.5,
            'tree_method': 'hist',
            'max_leaves': 64,
            'seed': 42
        }
        
        # Training parameters
        self.exploration_steps = 30000
        self.num_boost_round = 200
        self.gamma = 0.995
        self.reward_scale = 1e-6
        self.early_stopping_rounds = 10
        
        # Epsilon-greedy parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Track best scores
        self.best_eval_reward = float('-inf')
        self.best_eval_reward_dr = float('-inf')
        
        # Environment switching parameters
        self.dr_training_ratio = 0.3
        self.current_env = None
        self.env_no_dr = TimeLimit(FastHIVPatient(domain_randomization=False), max_episode_steps=200)
        self.env_dr = TimeLimit(FastHIVPatient(domain_randomization=True), max_episode_steps=200)

    def act(self, observation, use_random=False):
        if use_random or random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            q_values = []
            obs_reshaped = observation.reshape(1, -1)
            for a in range(self.action_dim):
                if self.models[a] is not None:
                    obs_scaled = self.scalers[a].transform(obs_reshaped)
                    q_values.append(self.models[a].predict(xgb.DMatrix(obs_scaled))[0])
                else:
                    q_values.append(float('-inf'))
            action = np.argmax(q_values)
        
        self.epsilon = max(self.epsilon_end, 
                         self.epsilon * self.epsilon_decay)
        return action

    def collect_transitions(self, steps, use_random=True):
        transitions = []
        state, _ = env.reset()
        
        for _ in range(steps):
            action = self.act(state, use_random)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            transitions.append((state, action, reward, next_state, done))
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
                
        return transitions

    def _prepare_fqi_dataset(self, transitions):
        states = np.vstack([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.vstack([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])
        
        action_datasets = [[] for _ in range(self.action_dim)]
        action_targets = [[] for _ in range(self.action_dim)]
        
        scaled_rewards = rewards * self.reward_scale
        
        next_q_values = np.zeros((len(states), self.action_dim))
        if self.models[0] is not None:
            for a in range(self.action_dim):
                next_states_scaled = self.scalers[a].transform(next_states)
                next_q_values[:, a] = self.models[a].predict(xgb.DMatrix(next_states_scaled))
        
        max_next_q = np.max(next_q_values, axis=1)
        
        for i in range(len(states)):
            action = int(actions[i])
            target = scaled_rewards[i]
            if not dones[i]:
                target += self.gamma * max_next_q[i]
            
            action_datasets[action].append(states[i])
            action_targets[action].append(target)
        
        return action_datasets, action_targets

    def evaluate(self, env, num_episodes=5):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.act(state, use_random=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
        return total_reward / num_episodes

    def train(self, num_epochs=6, episodes_per_epoch=200):
        print("Starting initial exploration...")
        all_transitions = []
        
        # Collect initial transitions from both environments
        for env in [self.env_no_dr, self.env_dr]:
            self.current_env = env
            env_transitions = self.collect_transitions(
                self.exploration_steps // 2, 
                use_random=True
            )
            all_transitions.extend(env_transitions)
            
            if isinstance(env, TimeLimit) and hasattr(env, 'env') and hasattr(env.env, 'domain_randomization'):
                env_type = 'with DR' if env.env.domain_randomization else 'no DR'
                print(f"\nCollected {len(env_transitions)} transitions from environment {env_type}")
        
        all_rewards = []
        eval_rewards = []
        eval_rewards_dr = []
        running_avg = []
        
        print("\nStarting FQI training...")
        for epoch in range(num_epochs):
            epoch_rewards = []
            epoch_transitions = []
            
            # Collect episodes
            for episode in tqdm(range(episodes_per_epoch), 
                              desc=f"Epoch {epoch + 1}/{num_epochs}"):
                self.current_env = self.env_dr if random.random() < self.dr_training_ratio else self.env_no_dr
                
                episode_transitions = self.collect_transitions(200, use_random=False)  # max episode length
                episode_reward = sum(t[2] for t in episode_transitions)
                
                epoch_transitions.extend(episode_transitions)
                epoch_rewards.append(episode_reward)
                all_rewards.append(episode_reward)
            
            # Add new transitions to dataset
            all_transitions.extend(epoch_transitions)
            
            # Train on all data with bootstrapping
            n_samples = len(all_transitions)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_transitions = [all_transitions[i] for i in bootstrap_idx]
            
            action_datasets, action_targets = self._prepare_fqi_dataset(bootstrap_transitions)
            
            # Train models
            for a in range(self.action_dim):
                if len(action_datasets[a]) > 0:
                    X = np.array(action_datasets[a])
                    y = np.array(action_targets[a])
                    
                    X_scaled = self.scalers[a].fit_transform(X)
                    
                    split_idx = int(0.8 * len(X))
                    dtrain = xgb.DMatrix(X_scaled[:split_idx], label=y[:split_idx])
                    dval = xgb.DMatrix(X_scaled[split_idx:], label=y[split_idx:])
                    
                    self.models[a] = xgb.train(
                        self.xgb_params,
                        dtrain,
                        num_boost_round=self.num_boost_round,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=False
                    )
            
            # Evaluate and save checkpoint
            eval_reward = self.evaluate(self.env_no_dr, num_episodes=10)
            eval_reward_dr = self.evaluate(self.env_dr, num_episodes=10)
            
            eval_rewards.append(eval_reward)
            eval_rewards_dr.append(eval_reward_dr)
            running_avg.append(np.mean(epoch_rewards))
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average Reward: {np.mean(epoch_rewards):.2e}")
            print(f"Evaluation Reward (no DR): {eval_reward:.2e}")
            print(f"Evaluation Reward (with DR): {eval_reward_dr:.2e}")
            
            self.save_checkpoint(epoch, eval_reward, eval_reward_dr)
        
        return all_rewards, eval_rewards, eval_rewards_dr, running_avg
    
    def plot_training_progress(self, rewards, eval_rewards, eval_rewards_dr, running_avg):
        plt.figure(figsize=(15, 12))
        
        plt.subplot(4, 1, 1)
        plt.plot(rewards)
        plt.title("Episode Rewards")
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(running_avg)
        plt.title("Running Average Reward")
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(eval_rewards, 'r-', label='No DR')
        plt.plot(eval_rewards_dr, 'b-', label='With DR')
        plt.title("Evaluation Rewards")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot([max(eval_rewards[:i+1]) for i in range(len(eval_rewards))], 'r-', label='No DR')
        plt.plot([max(eval_rewards_dr[:i+1]) for i in range(len(eval_rewards_dr))], 'b-', label='With DR')
        plt.title("Best Evaluation Rewards")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("fqi_detailed_progress2.png")
        plt.show()

    def save(self, path, save_dict=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        base_dict = {
            'models': self.models,
            'scalers': self.scalers,
            'epsilon': self.epsilon
        }
        
        if save_dict:
            base_dict.update(save_dict)
            
        joblib.dump(base_dict, path)
        
    def save_checkpoint(self, epoch, eval_reward, eval_reward_dr):
        checkpoint_dir = "trained_models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_dict = {
            'epoch': epoch,
            'eval_reward': eval_reward,
            'eval_reward_dr': eval_reward_dr,
            'timestamp': time.time()
        }
        
        # Save checkpoint
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        self.save(checkpoint_path, save_dict)
        
        # Save best models if we have new bests
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.save("trained_models/best_model_no_dr.pt", 
                     {'eval_reward': eval_reward})
            
        if eval_reward_dr > self.best_eval_reward_dr:
            self.best_eval_reward_dr = eval_reward_dr
            self.save("trained_models/best_model_dr.pt", 
                     {'eval_reward_dr': eval_reward_dr})
    
    def load(self, path=None):
        if path is None:
            path = os.getcwd() + "/trained_models/best_model_fqi2.pt"
            
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return False
            
        try:
            save_dict = joblib.load(path)
            self.models = save_dict['models']
            self.scalers = save_dict['scalers']
            if 'epsilon' in save_dict:
                self.epsilon = save_dict['epsilon']
                
            print(f"Successfully loaded model from {path}")
            if 'eval_reward' in save_dict:
                print(f"Regular env score: {save_dict['eval_reward']:.2e}")
            if 'eval_reward_dr' in save_dict:
                print(f"DR env score: {save_dict['eval_reward_dr']:.2e}")
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Initialize agent
    agent = ProjectAgent()
    
    # Train with enhanced parameters
    rewards, eval_rewards, eval_rewards_dr, running_avg = agent.train(
        num_epochs=100,
        episodes_per_epoch=30
    )
    
    # Plot training progress
    agent.plot_training_progress(rewards, eval_rewards, eval_rewards_dr, running_avg)
