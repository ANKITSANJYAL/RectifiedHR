#!/usr/bin/env python3
"""
Energy profiling for RectifiedHR research.
Analyzes latent energy evolution across denoising steps for different CFG scales and models.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple

def load_latent_data(data_dir: str) -> Dict:
    """Load latent energy data from JSON files."""
    data = {}
    pattern = Path(data_dir) / "latents_*.json"
    
    for file_path in glob.glob(str(pattern)):
        with open(file_path, 'r') as f:
            latent_data = json.load(f)
            
        metadata = latent_data['metadata']
        key = f"{metadata['sampler']}_cfg{metadata['cfg_scale']}"
        
        data[key] = {
            'energies': np.array(latent_data['energies']),
            'timesteps': np.array(latent_data['timesteps']),
            'metadata': metadata
        }
    
    return data

def plot_energy_trends(data: Dict, output_dir: str = "experiments/energy_plots"):
    """Plot energy trends for different CFG scales and samplers."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Latent Energy Evolution Across Denoising Steps', fontsize=16, fontweight='bold')
    
    # Plot 1: Energy vs Timestep for different CFG scales
    ax1 = axes[0, 0]
    cfg_scales = sorted(set([data[key]['metadata']['cfg_scale'] for key in data.keys()]))
    
    for cfg_scale in cfg_scales:
        # Find data for this CFG scale (assuming DDIM sampler)
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            timesteps = data[key]['timesteps']
            ax1.plot(timesteps, energies, label=f'CFG={cfg_scale}', linewidth=2)
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Energy (||latent||² / num_elements)')
    ax1.set_title('Energy Evolution vs CFG Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy vs Timestep for different samplers
    ax2 = axes[0, 1]
    samplers = sorted(set([data[key]['metadata']['sampler'] for key in data.keys()]))
    
    for sampler in samplers:
        # Find data for this sampler (assuming CFG=7)
        key = f"{sampler}_cfg7"
        if key in data:
            energies = data[key]['energies']
            timesteps = data[key]['timesteps']
            ax2.plot(timesteps, energies, label=f'{sampler.upper()}', linewidth=2)
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Energy (||latent||² / num_elements)')
    ax2.set_title('Energy Evolution vs Sampler')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final energy vs CFG scale
    ax3 = axes[1, 0]
    final_energies = []
    cfg_values = []
    
    for key in data.keys():
        if 'ddim' in key:  # Focus on DDIM for this plot
            final_energies.append(data[key]['energies'][-1])
            cfg_values.append(data[key]['metadata']['cfg_scale'])
    
    ax3.scatter(cfg_values, final_energies, s=100, alpha=0.7)
    ax3.set_xlabel('CFG Scale')
    ax3.set_ylabel('Final Energy')
    ax3.set_title('Final Energy vs CFG Scale')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy stability (variance across timesteps)
    ax4 = axes[1, 1]
    energy_variances = []
    cfg_values_var = []
    
    for key in data.keys():
        if 'ddim' in key:
            energy_variances.append(np.var(data[key]['energies']))
            cfg_values_var.append(data[key]['metadata']['cfg_scale'])
    
    ax4.scatter(cfg_values_var, energy_variances, s=100, alpha=0.7, color='orange')
    ax4.set_xlabel('CFG Scale')
    ax4.set_ylabel('Energy Variance')
    ax4.set_title('Energy Stability vs CFG Scale')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'energy_trends.png'

def plot_energy_heatmap(data: Dict, output_dir: str = "experiments/energy_plots"):
    """Create heatmap showing energy evolution across timesteps and CFG scales."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for heatmap
    cfg_scales = sorted(set([data[key]['metadata']['cfg_scale'] for key in data.keys() if 'ddim' in key]))
    
    # Create energy matrix
    max_timesteps = max(len(data[key]['energies']) for key in data.keys() if 'ddim' in key)
    energy_matrix = np.zeros((len(cfg_scales), max_timesteps))
    
    for i, cfg_scale in enumerate(cfg_scales):
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            energy_matrix[i, :len(energies)] = energies
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(energy_matrix, 
                xticklabels=[str(x) for x in range(0, max_timesteps, 5)],
                yticklabels=cfg_scales,
                cmap='viridis',
                cbar_kws={'label': 'Energy'})
    
    plt.xlabel('Timestep')
    plt.ylabel('CFG Scale')
    plt.title('Energy Evolution Heatmap')
    plt.tight_layout()
    plt.savefig(output_path / 'energy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'energy_heatmap.png'

def create_comprehensive_energy_analysis(data: Dict, output_dir: str = "experiments/energy_plots"):
    """Create comprehensive energy analysis for research paper."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up publication-ready style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 300
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Energy Analysis for RectifiedHR Research', fontsize=18, fontweight='bold')
    
    # Plot 1: Energy evolution comparison (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    
    cfg_scales = sorted(set([data[key]['metadata']['cfg_scale'] for key in data.keys() if 'ddim' in key]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg_scales)))
    
    for i, cfg_scale in enumerate(cfg_scales):
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            timesteps = data[key]['timesteps']
            ax1.plot(timesteps, energies, label=f'CFG={cfg_scale}', 
                    color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Energy (||latent||² / num_elements)')
    ax1.set_title('Energy Evolution Across CFG Scales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy gradient analysis (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    for i, cfg_scale in enumerate(cfg_scales):
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            timesteps = data[key]['timesteps']
            gradients = np.gradient(energies)
            ax2.plot(timesteps, gradients, label=f'CFG={cfg_scale}', 
                    color=colors[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Energy Gradient')
    ax2.set_title('Energy Stability Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final energy vs CFG scale (bottom left)
    ax3 = fig.add_subplot(gs[1:, :2])
    
    final_energies = []
    cfg_values = []
    
    for key in data.keys():
        if 'ddim' in key:
            final_energies.append(data[key]['energies'][-1])
            cfg_values.append(data[key]['metadata']['cfg_scale'])
    
    if final_energies:
        ax3.scatter(cfg_values, final_energies, s=150, alpha=0.7, color='red')
        ax3.set_xlabel('CFG Scale')
        ax3.set_ylabel('Final Energy')
        ax3.set_title('Final Energy vs CFG Scale')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(cfg_values) > 1:
            z = np.polyfit(cfg_values, final_energies, 1)
            p = np.poly1d(z)
            ax3.plot(cfg_values, p(cfg_values), "r--", alpha=0.8, linewidth=2)
    
    # Plot 4: Energy statistics summary (bottom right)
    ax4 = fig.add_subplot(gs[1:, 2:])
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = "Energy Analysis Summary:\n\n"
    
    for cfg_scale in cfg_scales:
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            stats_text += f"CFG {cfg_scale}:\n"
            stats_text += f"  Mean: {np.mean(energies):.4f}\n"
            stats_text += f"  Std: {np.std(energies):.4f}\n"
            stats_text += f"  Final: {energies[-1]:.4f}\n"
            stats_text += f"  Variance: {np.var(energies):.4f}\n\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.savefig(output_path / 'comprehensive_energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'comprehensive_energy_analysis.png'

def create_publication_figures(data: Dict, output_dir: str = "experiments/energy_plots"):
    """Create publication-ready figures for research paper."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Energy evolution (publication quality)
    plt.figure(figsize=(10, 6))
    
    cfg_scales = sorted(set([data[key]['metadata']['cfg_scale'] for key in data.keys() if 'ddim' in key]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg_scales)))
    
    for i, cfg_scale in enumerate(cfg_scales):
        key = f"ddim_cfg{cfg_scale}"
        if key in data:
            energies = data[key]['energies']
            timesteps = data[key]['timesteps']
            plt.plot(timesteps, energies, label=f'CFG={cfg_scale}', 
                    color=colors[i], linewidth=2.5, alpha=0.9)
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Latent Energy', fontsize=14)
    plt.title('Energy Evolution Across Denoising Steps', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_energy_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Energy stability analysis
    plt.figure(figsize=(10, 6))
    
    stability_metrics = []
    cfg_values = []
    
    for key in data.keys():
        if 'ddim' in key:
            energies = data[key]['energies']
            # Calculate stability as inverse of variance
            stability = 1.0 / (np.var(energies) + 1e-6)
            stability_metrics.append(stability)
            cfg_values.append(data[key]['metadata']['cfg_scale'])
    
    if stability_metrics:
        plt.bar(cfg_values, stability_metrics, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        plt.xlabel('CFG Scale', fontsize=14)
        plt.ylabel('Energy Stability (1/Variance)', fontsize=14)
        plt.title('Energy Stability Analysis', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path / 'figure2_energy_stability.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Figure 3: Final energy correlation
    plt.figure(figsize=(10, 6))
    
    final_energies = []
    cfg_values = []
    
    for key in data.keys():
        if 'ddim' in key:
            final_energies.append(data[key]['energies'][-1])
            cfg_values.append(data[key]['metadata']['cfg_scale'])
    
    if final_energies:
        plt.scatter(cfg_values, final_energies, s=100, alpha=0.7, color='red', edgecolors='darkred', linewidth=1)
        plt.xlabel('CFG Scale', fontsize=14)
        plt.ylabel('Final Energy', fontsize=14)
        plt.title('Final Energy vs CFG Scale', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(cfg_values) > 1:
            correlation = np.corrcoef(cfg_values, final_energies)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure3_final_energy_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_energy_statistics(data: Dict) -> Dict:
    """Analyze energy statistics for different configurations."""
    
    stats = {}
    
    for key, value in data.items():
        energies = value['energies']
        metadata = value['metadata']
        
        stats[key] = {
            'cfg_scale': metadata['cfg_scale'],
            'sampler': metadata['sampler'],
            'resolution': metadata['resolution'],
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'final_energy': energies[-1],
            'energy_variance': np.var(energies),
            'energy_range': np.max(energies) - np.min(energies)
        }
    
    return stats

def save_energy_analysis(stats: Dict, output_dir: str = "experiments/energy_plots"):
    """Save energy analysis results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats_file = output_path / 'energy_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Create summary table
    summary_data = []
    for key, stat in stats.items():
        summary_data.append({
            'Configuration': key,
            'CFG Scale': stat['cfg_scale'],
            'Sampler': stat['sampler'],
            'Mean Energy': f"{stat['mean_energy']:.4f}",
            'Std Energy': f"{stat['std_energy']:.4f}",
            'Final Energy': f"{stat['final_energy']:.4f}",
            'Energy Variance': f"{stat['energy_variance']:.4f}"
        })
    
    # Save summary as CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    summary_file = output_path / 'energy_summary.csv'
    df.to_csv(summary_file, index=False)
    
    print(f"Energy analysis saved to: {output_path}")
    print(f"Statistics: {stats_file}")
    print(f"Summary: {summary_file}")
    
    return stats_file, summary_file

def main():
    parser = argparse.ArgumentParser(description="Energy profiling for RectifiedHR research")
    parser.add_argument("--data_dir", type=str, default="experiments/baseline", 
                       help="Directory containing latent data files")
    parser.add_argument("--output_dir", type=str, default="experiments/energy_plots", 
                       help="Output directory for plots and analysis")
    parser.add_argument("--plot_heatmap", action="store_true", 
                       help="Generate energy heatmap")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Generate comprehensive analysis for research paper")
    parser.add_argument("--publication", action="store_true",
                       help="Generate publication-ready figures")
    
    args = parser.parse_args()
    
    # Load latent data
    print(f"Loading latent data from: {args.data_dir}")
    data = load_latent_data(args.data_dir)
    
    if not data:
        print("No latent data found. Please run generate.py first.")
        return
    
    print(f"Loaded {len(data)} configurations:")
    for key in data.keys():
        print(f"  - {key}")
    
    # Generate plots
    print("\nGenerating energy trend plots...")
    plot_energy_trends(data, args.output_dir)
    
    if args.plot_heatmap:
        print("Generating energy heatmap...")
        plot_energy_heatmap(data, args.output_dir)
    
    if args.comprehensive:
        print("Generating comprehensive energy analysis...")
        create_comprehensive_energy_analysis(data, args.output_dir)
    
    if args.publication:
        print("Generating publication-ready figures...")
        create_publication_figures(data, args.output_dir)
    
    # Analyze statistics
    print("Analyzing energy statistics...")
    stats = analyze_energy_statistics(data)
    save_energy_analysis(stats, args.output_dir)
    
    print("\nEnergy profiling complete!")

if __name__ == "__main__":
    main() 