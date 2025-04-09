import random
import os
import numpy as np
import torch
import glob
import pandas as pd
from tqdm import tqdm

from evaluate import evaluate_HIV, evaluate_HIV_population
from train2 import ProjectAgent

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# Define the reward thresholds from grading.py
REWARD_THRESHOLDS = [
    3432807.680391572,
    1e8,
    1e9,
    1e10,
    2e10,
    5e10
]

REWARD_DR_THRESHOLDS = [
    1e10,
    2e10,
    5e10
]

def evaluate_checkpoint(agent, checkpoint_path):
    # Load the checkpoint
    if not agent.load(checkpoint_path):
        return None
    
    seed_everything(seed=42)

    # Get the checkpoint number from the filename
    checkpoint_num = int(checkpoint_path.split('_')[-1].split('.')[0])
    
    # Evaluate on both environments
    score_agent = evaluate_HIV(agent=agent, nb_episode=5)
    score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=20)
    
    # Check which thresholds are passed
    regular_passes = [score_agent >= threshold for threshold in REWARD_THRESHOLDS]
    dr_passes = [score_agent_dr >= threshold for threshold in REWARD_DR_THRESHOLDS]
    
    return {
        'checkpoint': checkpoint_num,
        'path': checkpoint_path,
        'regular_score': score_agent,
        'dr_score': score_agent_dr,
        'regular_passes': sum(regular_passes),
        'dr_passes': sum(dr_passes),
        'total_passes': sum(regular_passes) + sum(dr_passes),
        'passes_all': all(regular_passes) and all(dr_passes)
    }

def find_best_checkpoint():
    seed_everything(seed=42)
    agent = ProjectAgent()
    
    # Get all checkpoint files
    checkpoint_dir = "trained_models/checkpoints"
    checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pt")
    
    if not checkpoint_files:
        print("No checkpoints found!")
        return
    
    results = []
    print("\nEvaluating checkpoints...")
    for checkpoint in tqdm(checkpoint_files):
        result = evaluate_checkpoint(agent, checkpoint)
        if result is not None:
            results.append(result)
    
    # Convert results to DataFrame for better analysis
    df = pd.DataFrame(results)
    
    # Sort by total passes and then by average score
    df['avg_score'] = (df['regular_score'] + df['dr_score']) / 2
    df = df.sort_values(['total_passes', 'avg_score'], ascending=[False, False])
    
    # Print summary
    print("\nCheckpoint Evaluation Summary:")
    print(f"Total checkpoints evaluated: {len(df)}")
    print(f"\nBest performing checkpoints:")
    print(df.head().to_string(index=False))
    
    # Find first checkpoint that passes all tests
    passing_checkpoints = df[df['passes_all']]
    if not passing_checkpoints.empty:
        best_checkpoint = passing_checkpoints.iloc[0]
        print(f"\nFound checkpoint that passes all tests!")
        print(f"Checkpoint: {best_checkpoint['checkpoint']}")
        print(f"Regular Score: {best_checkpoint['regular_score']:.2e}")
        print(f"DR Score: {best_checkpoint['dr_score']:.2e}")
        
        # Save the scores to score.txt
        with open("score.txt", "w") as f:
            f.write(f"{best_checkpoint['regular_score']}\n{best_checkpoint['dr_score']}")
            
        return best_checkpoint['path']
    else:
        print("\nNo checkpoint found that passes all tests.")
        print("\nBest checkpoint details:")
        best = df.iloc[0]
        print(f"Checkpoint {best['checkpoint']}:")
        print(f"Regular Score: {best['regular_score']:.2e} (passes {best['regular_passes']}/{len(REWARD_THRESHOLDS)})")
        print(f"DR Score: {best['dr_score']:.2e} (passes {best['dr_passes']}/{len(REWARD_DR_THRESHOLDS)})")
        return None

def main():
    seed_everything(seed=42)

    # Find and evaluate best checkpoint
    best_checkpoint = find_best_checkpoint()
    
    if best_checkpoint:
        print(f"\nBest checkpoint saved to score.txt: {best_checkpoint}")
    else:
        print("\nNo checkpoint fully passes all tests. Check the evaluation summary above.")

if __name__ == "__main__":
    seed_everything(seed=42)
    main()
