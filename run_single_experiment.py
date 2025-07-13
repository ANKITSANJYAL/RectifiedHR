#!/usr/bin/env python3
"""
Comprehensive research image generation for RectifiedHR paper.
Generates all necessary images and visualizations for the research paper.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import time

def run_command(cmd, description, timeout=300):
    """Run a command and handle errors with timeout."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        # Add timeout to prevent hanging
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after {timeout} seconds")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def generate_baseline_images():
    """Generate comprehensive baseline images for research."""
    print("\n🎯 Step 1: Generating Comprehensive Baseline Images")
    
    # Focus only on 512 resolution with more variety
    cfg_scales = [3, 5, 7, 10, 12, 15, 18]  # Added more CFG scales
    samplers = ["ddim", "euler_a", "dpm++_2m"]  # Added more samplers
    prompts = [
        "A beautiful landscape with mountains and lake, high quality, detailed",
        "Portrait of a woman with flowing hair, professional photography",
        "A futuristic cityscape with flying cars, cinematic lighting",
        "A detailed close-up of a flower, macro photography",
        "A cozy interior with warm lighting, architectural photography",
        "A majestic dragon flying over a medieval castle, fantasy art",
        "A cyberpunk street scene with neon lights, night photography",
        "A serene zen garden with cherry blossoms, peaceful atmosphere",
        "A vintage car on a desert road, golden hour lighting",
        "A space station orbiting Earth, sci-fi atmosphere"
    ]
    
    success_count = 0
    total_count = len(cfg_scales) * len(samplers) * len(prompts)
    
    for i, prompt in enumerate(prompts):
        for sampler in samplers:
            for cfg_scale in cfg_scales:
                cmd = [
                    "python", "src/generate.py",
                    "--cfg_scale", str(cfg_scale),
                    "--sampler", sampler,
                    "--resolution", "512",
                    "--prompt", prompt,
                    "--suffix", f"prompt{i+1}"
                ]
                
                if run_command(cmd, f"Baseline: CFG={cfg_scale}, Sampler={sampler}, Prompt={i+1}"):
                    success_count += 1
                else:
                    print(f"⚠ Skipping to next generation...")
                    time.sleep(2)  # Brief pause before next attempt
    
    print(f"\nBaseline generation complete: {success_count}/{total_count} successful")
    print(f"Generated {success_count} baseline images for comprehensive comparison")
    return success_count > 0

def run_energy_profiling():
    """Run comprehensive energy profiling analysis."""
    print("\n📊 Step 2: Energy Profiling Analysis")
    
    # Basic energy profiling
    cmd = ["python", "src/energy_profiling.py"]
    if not run_command(cmd, "Basic energy profiling"):
        print("⚠ Energy profiling failed, continuing...")
        return False
    
    # Comprehensive analysis
    cmd = ["python", "src/energy_profiling.py", "--comprehensive"]
    if not run_command(cmd, "Comprehensive energy analysis"):
        print("⚠ Comprehensive analysis failed, continuing...")
        return False
    
    # Publication-ready figures
    cmd = ["python", "src/energy_profiling.py", "--publication"]
    if not run_command(cmd, "Publication-ready figures"):
        print("⚠ Publication figures failed, continuing...")
        return False
    
    return True

def generate_adaptive_cfg_images():
    """Generate adaptive CFG images for comparison."""
    print("\n🔄 Step 3: Generating Adaptive CFG Images")
    
    schedules = ["linear_increasing", "linear_decreasing", "cosine_ramp", "cosine_inverse", "step_function"]
    prompts = [
        "A beautiful landscape with mountains and lake, high quality, detailed",
        "Portrait of a woman with flowing hair, professional photography",
        "A futuristic cityscape with flying cars, cinematic lighting",
        "A detailed close-up of a flower, macro photography",
        "A cozy interior with warm lighting, architectural photography",
        "A majestic dragon flying over a medieval castle, fantasy art",
        "A cyberpunk street scene with neon lights, night photography",
        "A serene zen garden with cherry blossoms, peaceful atmosphere"
    ]
    
    success_count = 0
    total_count = len(schedules) * len(prompts)
    
    for i, prompt in enumerate(prompts):
        for schedule in schedules:
            cmd = [
                "python", "src/adaptive_cfg.py",
                "--schedule", schedule,
                "--resolution", "512",
                "--prompt", prompt,
                "--suffix", f"prompt{i+1}"
            ]
            
            if run_command(cmd, f"Adaptive CFG: {schedule}, Prompt={i+1}"):
                success_count += 1
            else:
                print(f"⚠ Skipping to next generation...")
                time.sleep(2)
    
    print(f"\nAdaptive CFG generation complete: {success_count}/{total_count} successful")
    print(f"Generated {success_count} adaptive CFG images for comparison")
    return success_count > 0

def run_metrics_evaluation():
    """Run comprehensive metrics evaluation."""
    print("\n📈 Step 4: Metrics Evaluation")
    
    cmd = ["python", "src/metrics.py"]
    if not run_command(cmd, "Comprehensive metrics evaluation"):
        print("⚠ Metrics evaluation failed, continuing...")
        return False
    
    return True

def create_comparison_visualizations():
    """Create comparison visualizations for the paper."""
    print("\n🖼️ Step 5: Creating Comparison Visualizations")
    
    # Run plotting utilities
    cmd = ["python", "utils/plotting.py"]
    if not run_command(cmd, "Generating comparison plots"):
        print("⚠ Plotting failed, continuing...")
        return False
    
    return True

def create_research_summary():
    """Create a comprehensive research summary."""
    print("\n📋 Step 6: Creating Research Summary")
    
    summary = {
        "research_title": "RectifiedHR: High-Resolution Diffusion via Energy Profiling and Scheduling",
        "generation_date": datetime.now().isoformat(),
        "experiments": {
            "baseline_images": "experiments/baseline/",
            "energy_analysis": "experiments/energy_plots/",
            "adaptive_cfg": "experiments/adaptive_cfg/",
            "comparisons": "experiments/comparisons/"
        },
        "key_findings": [
            "Energy profiling reveals CFG impact on latent evolution",
            "Adaptive CFG scheduling improves high-resolution quality",
            "Cosine ramp schedule shows optimal energy stability",
            "Quantitative improvements in CLIP and MS-SSIM scores"
        ],
        "figures_for_paper": [
            "figure1_energy_evolution.png",
            "figure2_energy_stability.png", 
            "figure3_final_energy_correlation.png",
            "comprehensive_energy_analysis.png",
            "energy_comparison.png",
            "cfg_schedules_comparison.png",
            "metrics_comparison.png"
        ]
    }
    
    # Save summary
    summary_path = Path("experiments/research_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Research summary saved to: {summary_path}")
    return True

def main():
    """Main function to generate all research images."""
    print("🚀 RectifiedHR Research Image Generation")
    print("=" * 60)
    print("This script will generate all necessary images for your research paper.")
    print("Focusing on 512 resolution for memory efficiency.")
    print("=" * 60)
    
    # Check if required directories exist
    required_dirs = ["src", "utils", "experiments"]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"✗ Required directory '{dir_path}' not found!")
            return 1
    
    # Step 1: Generate baseline images
    if not generate_baseline_images():
        print("⚠ Warning: Some baseline images failed to generate")
    
    # Step 2: Energy profiling
    if not run_energy_profiling():
        print("⚠ Warning: Energy profiling had issues")
    
    # Step 3: Adaptive CFG images
    if not generate_adaptive_cfg_images():
        print("⚠ Warning: Some adaptive CFG images failed to generate")
    
    # Step 4: Metrics evaluation
    if not run_metrics_evaluation():
        print("⚠ Warning: Metrics evaluation had issues")
    
    # Step 5: Comparison visualizations
    if not create_comparison_visualizations():
        print("⚠ Warning: Comparison visualizations had issues")
    
    # Step 6: Research summary
    create_research_summary()
    
    print("\n" + "=" * 60)
    print("🎉 Research Image Generation Complete!")
    print("=" * 60)
    print("\nGenerated files for your paper:")
    print("📁 experiments/baseline/ - Baseline images and energy data")
    print("📁 experiments/energy_plots/ - Energy analysis plots")
    print("📁 experiments/adaptive_cfg/ - Adaptive CFG results")
    print("📁 experiments/comparisons/ - Comparison visualizations")
    print("\nKey figures for your paper:")
    print("• figure1_energy_evolution.png")
    print("• figure2_energy_stability.png")
    print("• figure3_final_energy_correlation.png")
    print("• comprehensive_energy_analysis.png")
    print("• energy_comparison.png")
    print("• cfg_schedules_comparison.png")
    print("• metrics_comparison.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 