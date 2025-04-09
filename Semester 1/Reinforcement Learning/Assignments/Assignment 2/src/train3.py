import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from agent import DQNAgent

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def main():
    # Create directories
    experiment_name = f"hiv_treatment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = "results"
    log_dir = os.path.join(base_dir, experiment_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    img_dir = os.path.join(log_dir, "images")
    
    # Create directories if they don't exist
    for dir_path in [log_dir, ckpt_dir, img_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Setup logging and tensorboard
    setup_logging(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Training parameters
    params = {
        # General parameters
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "load_ckpt": False,  # Set to True if you want to load from a checkpoint
        "writer": writer,
        
        # Environment parameters
        "max_days": 200,
        "treatment_days": 1,
        "reward_scaler": 1e+8,
        
        # Training parameters
        "memory_size": int(1e6),
        "batch_size": 2048,
        "lr": 2e-4,
        "l2_reg": 0.0,
        "grad_clip": 1000.0,
        "target_update": 3000,
        "max_epsilon": 1.0,
        "min_epsilon": 0.05,
        "epsilon_decay": 1/200,
        "decay_option": 'logistic',
        "discount_factor": 0.99,
        "n_train": 1,
        
        # Network parameters
        "hidden_dim": 1024,
        
        # PER parameters
        "per": True,
        "alpha": 0.2,
        "beta": 0.6,
        "beta_increment_per_sampling": 0.000005,
        "prior_eps": 1e-6,
        
        # Double DQN
        "double_dqn": True,
    }

    # Create and train agent
    agent = DQNAgent(**params)
    
    try:
        agent.train(
            max_episodes=1000,      # Total number of episodes to train
            log_freq=10,            # How often to log training stats
            test_freq=50,           # How often to run test episodes
            save_freq=100,          # How often to save checkpoints
            img_dir=img_dir         # Where to save visualizations
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        # Save final checkpoint
        agent.save_ckpt(
            episode=-1,  # -1 indicates final checkpoint
            path=os.path.join(ckpt_dir, 'final_ckpt.pt')
        )
        writer.close()

if __name__ == "__main__":
    main()
