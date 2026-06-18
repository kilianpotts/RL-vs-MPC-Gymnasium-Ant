import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_and_smooth_data(csv_path, window_size=20):
    """
    Loads a specific CSV file, extracts timesteps and rewards,
    and applies a rolling window average to smooth the curves.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
        
    df.columns = [col.strip().lower() for col in df.columns]
    
    if 'timestep' not in df.columns or 'episode_reward' not in df.columns:
        print(f"Error: CSV at {csv_path} must contain 'timestep' and 'episode_reward' columns.")
        return None
        
    df['reward_smooth'] = df['episode_reward'].rolling(window=window_size, min_periods=1).mean()
    return df[['timestep', 'reward_smooth']]

def plot_expert_comparison(sac_path, ppo_path, output_path="expert_comparison.pdf"):
    """
    Generates a scaled, clean comparison plot from the target files.
    """
    plt.figure(figsize=(3.5, 2.8))

    plt.rcParams['font.weight'] = 'medium'
    plt.rcParams['axes.labelweight'] = 'medium'
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.labelsize'] = 10.5
    plt.rcParams['legend.fontsize'] = 9.5
    
    if sac_path:
        sac_data = load_and_smooth_data(sac_path)
        if sac_data is not None:
            plt.plot(sac_data['timestep'], sac_data['reward_smooth'], 
                     label='SAC (Off-Policy)', color='#1f77b4', linewidth=1)
                 
    if ppo_path:
        ppo_data = load_and_smooth_data(ppo_path)
        if ppo_data is not None:
            plt.plot(ppo_data['timestep'], ppo_data['reward_smooth'], 
                     label='PPO (On-Policy)', color='#ff7f0e', linewidth=1)
    
    plt.xlabel('Total Environment Steps', labelpad=10)
    plt.ylabel('Smoothed Episode Reward')
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    plt.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Figure successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training performance metrics between SAC and PPO controllers.")
    parser.add_argument("--sac", type=str, help="Path to the SAC training curve CSV file.")
    parser.add_argument("--ppo", type=str, help="Path to the PPO training curve CSV file.")
    parser.add_argument("--output", type=str, default="/workspaces/gymnasium_ws/evaluation/plots/forward_expert_comparison.pdf", help="Output destination for the PDF plot.")
    
    args = parser.parse_args()
    
    if not args.sac and not args.ppo:
        parser.error("You must provide at least one file path via --sac or --ppo.")
        
    plot_expert_comparison(sac_path=args.sac, ppo_path=args.ppo, output_path=args.output)