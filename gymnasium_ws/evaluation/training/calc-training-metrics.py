import argparse
import glob
import os
import numpy as np
import pandas as pd

def calculate_metrics(csv_path, convergence_threshold_pct=0.80, smooth_window=20):
    """Parses a training curve CSV and calculates AUC, Convergence Step, and Asymptotic Reward."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
        
    df.columns = [col.strip().lower() for col in df.columns]
    if 'timestep' not in df.columns or 'episode_reward' not in df.columns:
        return None
        
    df = df.sort_values(by='timestep').reset_index(drop=True)
    x = df['timestep'].values
    y = df['episode_reward'].values
    
    total_steps = x[-1] - x[0]
    normalized_auc = np.trapezoid(y, x) / total_steps if total_steps > 0 else 0.0
    
    asymptotic_cutoff = int(len(y) * 0.90)
    asymptotic_reward = np.mean(y[asymptotic_cutoff:])
    
    smoothed_y = df['episode_reward'].rolling(window=smooth_window, min_periods=1).mean().values
    max_reward = np.max(smoothed_y)
    min_reward = np.min(smoothed_y)
    target_threshold = min_reward + (convergence_threshold_pct * (max_reward - min_reward))
    
    convergence_step = None
    for i in range(len(smoothed_y)):
        if np.all(smoothed_y[i:] >= target_threshold):
            convergence_step = int(x[i])
            break
            
    return {
        "auc": normalized_auc,
        "convergence_step": convergence_step,
        "asymptotic_reward": asymptotic_reward
    }

def collect_directory_metrics(dir_path, threshold):
    """Scans directory and returns a dictionary of skill -> metrics mappings."""
    if not dir_path or not os.path.isdir(dir_path):
        return {}
    csv_files = glob.glob(os.path.join(dir_path, "training_*.csv"))
    data = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        skill = parts[1] if len(parts) > 1 else "unknown"
        metrics = calculate_metrics(file_path, convergence_threshold_pct=threshold)
        if metrics:
            data[skill] = metrics
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and compare DRL training directories.")
    parser.add_argument("--sac_dir", type=str, required=True, help="Directory containing SAC expert CSVs.")
    parser.add_argument("--ppo_dir", type=str, required=True, help="Directory containing PPO expert CSVs.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Convergence threshold percentage (0.0 - 1.0).")
    args = parser.parse_args()
    
    sac_data = collect_directory_metrics(args.sac_dir, args.threshold)
    ppo_data = collect_directory_metrics(args.ppo_dir, args.threshold)
    
    # 1. Print Standard Raw Metrics Table
    print(f"\n{'Algorithm (Skill)':<25} | {'Normalized AUC':<15} | {'Convergence Step':<16} | {'Asymptotic Reward':<18}")
    print("-" * 82)
    for skill in sorted(set(list(sac_data.keys()) + list(ppo_data.keys()))):
        if skill in sac_data:
            s = sac_data[skill]
            print(f"{f'SAC ({skill})':<25} | {s['auc']:<15.4f} | {str(s['convergence_step']):<16} | {s['asymptotic_reward']:<18.4f}")
        if skill in ppo_data:
            p = ppo_data[skill]
            print(f"{f'PPO ({skill})':<25} | {p['auc']:<15.4f} | {str(p['convergence_step']):<16} | {p['asymptotic_reward']:<18.4f}")
            
    # 2. Print Relative Differences Table (SAC vs PPO)
    print(f"\n{'Relative Comparison (SAC vs PPO)':<32} | {'AUC Delta %':<15} | {'Conv. Step Delta %':<20} | {'Asymptotic Delta %':<20}")
    print("-" * 94)
    
    auc_deltas, conv_deltas, asymp_deltas = [], [], []
    
    for skill in sorted(sac_data.keys()):
        if skill in ppo_data:
            sac = sac_data[skill]
            ppo = ppo_data[skill]
            
            auc_d = ((sac['auc'] - ppo['auc']) / abs(ppo['auc'])) * 100
            asymp_d = ((sac['asymptotic_reward'] - ppo['asymptotic_reward']) / abs(ppo['asymptotic_reward'])) * 100
            
            auc_deltas.append(auc_d)
            asymp_deltas.append(asymp_d)
            
            conv_str = "N/A"
            if sac['convergence_step'] is not None and ppo['convergence_step'] is not None:
                conv_d = ((sac['convergence_step'] - ppo['convergence_step']) / ppo['convergence_step']) * 100
                conv_deltas.append(conv_d)
                conv_str = f"{conv_d:+.2f}%"
                
            print(f"{f'SAC vs PPO ({skill})':<32} | {auc_d:+.2f}%{'':<9} | {conv_str:<20} | {asymp_d:+.2f}%")

    # 3. Print Global Summary Averages
    print("-" * 94)
    avg_auc_d = np.mean(auc_deltas) if auc_deltas else 0.0
    avg_conv_d = np.mean(conv_deltas) if conv_deltas else 0.0
    avg_asymp_d = np.mean(asymp_deltas) if asymp_deltas else 0.0
    
    print(f"{'MEAN RELATIVE DIFFERENCE':<32} | {avg_auc_d:+.2f}%{'':<9} | {f'{avg_conv_d:+.2f}%':<20} | {avg_asymp_d:+.2f}%")
    print(f"Notes: Positive values favor SAC. Negative Conv. Step favors SAC (faster convergence).\n")