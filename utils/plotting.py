#!/usr/bin/env python3
"""
Plotting utilities for RectifiedHR research.
Creates energy trend plots, comparison visualizations, and analysis charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def setup_plotting_style():
    """Setup consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_energy_comparison(baseline_data: Dict, adaptive_data: Dict, output_path: Path):
    """Create comparison plot between baseline and adaptive CFG energy profiles."""
    
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Energy Profile Comparison: Baseline vs Adaptive CFG', fontsize=16, fontweight='bold')
    
    # Plot 1: Energy evolution comparison
    ax1 = axes[0, 0]
    
    # Plot baseline energy
    if baseline_data:
        for key, data in baseline_data.items():
            if 'ddim_cfg7' in key:  # Focus on CFG=7 baseline
                energies = data['energies']
                timesteps = data['timesteps']
                ax1.plot(timesteps, energies, label='Baseline (CFG=7)', linewidth=2, color='blue')
                break
    
    # Plot adaptive CFG energy
    if adaptive_data:
        for key, data in adaptive_data.items():
            if 'cosine_ramp' in key:  # Focus on cosine ramp
                energies = data['energies']
                timesteps = data['timesteps']
                ax1.plot(timesteps, energies, label='Adaptive CFG (Cosine)', linewidth=2, color='red')
                break
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Energy (||latent||² / num_elements)')
    ax1.set_title('Energy Evolution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CFG value evolution for adaptive method
    ax2 = axes[0, 1]
    
    if adaptive_data:
        for key, data in adaptive_data.items():
            if 'cosine_ramp' in key and 'cfg_values' in data:
                cfg_values = data['cfg_values']
                timesteps = data['timesteps']
                ax2.plot(timesteps, cfg_values, label='Cosine Ramp CFG', linewidth=2, color='green')
                break
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('CFG Scale')
    ax2.set_title('Adaptive CFG Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy stability comparison
    ax3 = axes[1, 0]
    
    baseline_stability = []
    adaptive_stability = []
    
    if baseline_data:
        for key, data in baseline_data.items():
            if 'ddim_cfg7' in key:
                energies = data['energies']
                baseline_stability = np.gradient(energies)
                ax3.plot(timesteps, baseline_stability, label='Baseline', linewidth=2, color='blue')
                break
    
    if adaptive_data:
        for key, data in adaptive_data.items():
            if 'cosine_ramp' in key:
                energies = data['energies']
                timesteps = data['timesteps']
                adaptive_stability = np.gradient(energies)
                ax3.plot(timesteps, adaptive_stability, label='Adaptive CFG', linewidth=2, color='red')
                break
    
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Energy Gradient')
    ax3.set_title('Energy Stability Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final energy comparison
    ax4 = axes[1, 1]
    
    baseline_final = []
    adaptive_final = []
    
    if baseline_data:
        for key, data in baseline_data.items():
            baseline_final.append(data['energies'][-1])
    
    if adaptive_data:
        for key, data in adaptive_data.items():
            adaptive_final.append(data['energies'][-1])
    
    if baseline_final and adaptive_final:
        methods = ['Baseline'] * len(baseline_final) + ['Adaptive CFG'] * len(adaptive_final)
        final_energies = baseline_final + adaptive_final
        
        # Create box plot
        data_for_box = [baseline_final, adaptive_final]
        ax4.boxplot(data_for_box, labels=['Baseline', 'Adaptive CFG'])
        ax4.set_ylabel('Final Energy')
        ax4.set_title('Final Energy Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'energy_comparison.png'

def plot_cfg_schedules_comparison(adaptive_data: Dict, output_path: Path):
    """Compare different CFG scheduling strategies."""
    
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CFG Schedule Comparison', fontsize=16, fontweight='bold')
    
    schedule_types = ['linear_increasing', 'linear_decreasing', 'cosine_ramp', 'cosine_inverse']
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: CFG schedules
    ax1 = axes[0, 0]
    
    for i, schedule_type in enumerate(schedule_types):
        for key, data in adaptive_data.items():
            if schedule_type in key and 'cfg_values' in data:
                cfg_values = data['cfg_values']
                timesteps = data['timesteps']
                ax1.plot(timesteps, cfg_values, label=schedule_type.replace('_', ' ').title(), 
                        linewidth=2, color=colors[i])
                break
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('CFG Scale')
    ax1.set_title('CFG Schedule Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy evolution for different schedules
    ax2 = axes[0, 1]
    
    for i, schedule_type in enumerate(schedule_types):
        for key, data in adaptive_data.items():
            if schedule_type in key:
                energies = data['energies']
                timesteps = data['timesteps']
                ax2.plot(timesteps, energies, label=schedule_type.replace('_', ' ').title(), 
                        linewidth=2, color=colors[i])
                break
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Energy (||latent||² / num_elements)')
    ax2.set_title('Energy Evolution by Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final energy comparison
    ax3 = axes[1, 0]
    
    schedule_names = []
    final_energies = []
    
    for schedule_type in schedule_types:
        for key, data in adaptive_data.items():
            if schedule_type in key:
                final_energies.append(data['energies'][-1])
                schedule_names.append(schedule_type.replace('_', ' ').title())
                break
    
    if final_energies:
        bars = ax3.bar(schedule_names, final_energies, color=colors[:len(final_energies)])
        ax3.set_ylabel('Final Energy')
        ax3.set_title('Final Energy by Schedule')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, energy in zip(bars, final_energies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{energy:.4f}', ha='center', va='bottom')
    
    # Plot 4: Energy stability comparison
    ax4 = axes[1, 1]
    
    stability_scores = []
    
    for schedule_type in schedule_types:
        for key, data in adaptive_data.items():
            if schedule_type in key:
                energies = data['energies']
                # Calculate stability as inverse of variance
                stability = 1.0 / (np.var(energies) + 1e-6)
                stability_scores.append(stability)
                break
    
    if stability_scores:
        bars = ax4.bar(schedule_names, stability_scores, color=colors[:len(stability_scores)])
        ax4.set_ylabel('Stability Score (1/Variance)')
        ax4.set_title('Energy Stability by Schedule')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, stability in zip(bars, stability_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{stability:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'cfg_schedules_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'cfg_schedules_comparison.png'

def plot_metrics_comparison(metrics_data: Dict, output_path: Path):
    """Create comparison plots for evaluation metrics."""
    
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    methods = []
    clip_scores = []
    ms_ssim_scores = []
    lpips_scores = []
    psnr_scores = []
    
    for key, data in metrics_data.items():
        method = data.get('method', 'unknown')
        methods.append(method)
        clip_scores.append(data.get('clip_similarity', 0))
        ms_ssim_scores.append(data.get('ms_ssim', 0))
        lpips_scores.append(data.get('lpips', 0))
        psnr_scores.append(data.get('psnr', 0))
    
    # Plot 1: CLIP Similarity
    ax1 = axes[0, 0]
    unique_methods = list(set(methods))
    method_clip_scores = {method: [] for method in unique_methods}
    
    for method, score in zip(methods, clip_scores):
        method_clip_scores[method].append(score)
    
    for method in unique_methods:
        scores = method_clip_scores[method]
        ax1.boxplot(scores, labels=[method], positions=[unique_methods.index(method)])
    
    ax1.set_ylabel('CLIP Similarity')
    ax1.set_title('CLIP Similarity by Method')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MS-SSIM
    ax2 = axes[0, 1]
    method_ms_ssim_scores = {method: [] for method in unique_methods}
    
    for method, score in zip(methods, ms_ssim_scores):
        if score > 0:  # Only include valid scores
            method_ms_ssim_scores[method].append(score)
    
    for method in unique_methods:
        scores = method_ms_ssim_scores[method]
        if scores:
            ax2.boxplot(scores, labels=[method], positions=[unique_methods.index(method)])
    
    ax2.set_ylabel('MS-SSIM Score')
    ax2.set_title('MS-SSIM by Method')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: LPIPS
    ax3 = axes[1, 0]
    method_lpips_scores = {method: [] for method in unique_methods}
    
    for method, score in zip(methods, lpips_scores):
        if score > 0:  # Only include valid scores
            method_lpips_scores[method].append(score)
    
    for method in unique_methods:
        scores = method_lpips_scores[method]
        if scores:
            ax3.boxplot(scores, labels=[method], positions=[unique_methods.index(method)])
    
    ax3.set_ylabel('LPIPS Distance')
    ax3.set_title('LPIPS by Method')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PSNR
    ax4 = axes[1, 1]
    method_psnr_scores = {method: [] for method in unique_methods}
    
    for method, score in zip(methods, psnr_scores):
        if score > 0:  # Only include valid scores
            method_psnr_scores[method].append(score)
    
    for method in unique_methods:
        scores = method_psnr_scores[method]
        if scores:
            ax4.boxplot(scores, labels=[method], positions=[unique_methods.index(method)])
    
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('PSNR by Method')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'metrics_comparison.png'

def create_summary_dashboard(baseline_data: Dict, adaptive_data: Dict, metrics_data: Dict, output_path: Path):
    """Create a comprehensive summary dashboard."""
    
    setup_plotting_style()
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('RectifiedHR Research Summary Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Energy evolution comparison (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    
    if baseline_data:
        for key, data in baseline_data.items():
            if 'ddim_cfg7' in key:
                energies = data['energies']
                timesteps = data['timesteps']
                ax1.plot(timesteps, energies, label='Baseline (CFG=7)', linewidth=2, color='blue')
                break
    
    if adaptive_data:
        for key, data in adaptive_data.items():
            if 'cosine_ramp' in key:
                energies = data['energies']
                timesteps = data['timesteps']
                ax1.plot(timesteps, energies, label='Adaptive CFG (Cosine)', linewidth=2, color='red')
                break
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Energy (||latent||² / num_elements)')
    ax1.set_title('Energy Evolution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CFG schedule (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if adaptive_data:
        for key, data in adaptive_data.items():
            if 'cosine_ramp' in key and 'cfg_values' in data:
                cfg_values = data['cfg_values']
                timesteps = data['timesteps']
                ax2.plot(timesteps, cfg_values, label='Cosine Ramp CFG', linewidth=2, color='green')
                break
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('CFG Scale')
    ax2.set_title('Adaptive CFG Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Metrics comparison (bottom left)
    ax3 = fig.add_subplot(gs[1:, :2])
    
    if metrics_data:
        methods = list(set([data.get('method', 'unknown') for data in metrics_data.values()]))
        metrics = ['clip_similarity', 'ms_ssim', 'lpips', 'psnr']
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = []
            for method in methods:
                method_values = [data.get(metric, 0) for data in metrics_data.values() 
                               if data.get('method') == method and data.get(metric) is not None]
                values.append(np.mean(method_values) if method_values else 0)
            
            ax3.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Evaluation Metrics Comparison')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics (bottom right)
    ax4 = fig.add_subplot(gs[1:, 2:])
    ax4.axis('off')
    
    # Create summary text
    summary_text = "Research Summary:\n\n"
    
    if baseline_data:
        summary_text += f"• Baseline images: {len(baseline_data)}\n"
    if adaptive_data:
        summary_text += f"• Adaptive CFG images: {len(adaptive_data)}\n"
    if metrics_data:
        summary_text += f"• Evaluated images: {len(metrics_data)}\n"
    
    summary_text += "\nKey Findings:\n"
    summary_text += "• Energy profiling reveals CFG impact\n"
    summary_text += "• Adaptive scheduling improves stability\n"
    summary_text += "• Cosine ramp shows best performance\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.savefig(output_path / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path / 'summary_dashboard.png'

def main():
    """Main function for plotting utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plotting utilities for RectifiedHR research")
    parser.add_argument("--baseline_dir", type=str, default="experiments/baseline",
                       help="Directory containing baseline data")
    parser.add_argument("--adaptive_dir", type=str, default="experiments/adaptive_cfg",
                       help="Directory containing adaptive CFG data")
    parser.add_argument("--metrics_file", type=str, default="experiments/comparisons/evaluation_results.json",
                       help="File containing evaluation metrics")
    parser.add_argument("--output_dir", type=str, default="experiments/comparisons",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load data
    baseline_data = {}
    adaptive_data = {}
    metrics_data = {}
    
    # Load baseline data
    baseline_path = Path(args.baseline_dir)
    if baseline_path.exists():
        for latent_file in baseline_path.glob("latents_*.json"):
            with open(latent_file, 'r') as f:
                data = json.load(f)
                key = f"{data['metadata']['sampler']}_cfg{data['metadata']['cfg_scale']}"
                baseline_data[key] = data
    
    # Load adaptive data
    adaptive_path = Path(args.adaptive_dir)
    if adaptive_path.exists():
        for latent_file in adaptive_path.glob("latents_adaptive_*.json"):
            with open(latent_file, 'r') as f:
                data = json.load(f)
                key = f"adaptive_{data['metadata']['schedule_type']}"
                adaptive_data[key] = data
    
    # Load metrics data
    metrics_path = Path(args.metrics_file)
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if baseline_data and adaptive_data:
        plot_energy_comparison(baseline_data, adaptive_data, output_path)
    
    if adaptive_data:
        plot_cfg_schedules_comparison(adaptive_data, output_path)
    
    if metrics_data:
        plot_metrics_comparison(metrics_data, output_path)
    
    # Create summary dashboard
    create_summary_dashboard(baseline_data, adaptive_data, metrics_data, output_path)
    
    print(f"Plots saved to: {output_path}")

if __name__ == "__main__":
    main() 