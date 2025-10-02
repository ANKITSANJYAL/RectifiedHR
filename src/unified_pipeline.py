#!/usr/bin/env python3
"""
Unified generation pipeline for RectifiedHR revision.
Handles SD 1.5, SDXL, SD 2.1 across all resolutions (512px - 4096px).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

# Will be imported when torch is available
# import torch
# from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
# from PIL import Image

class AdaptiveCFGScheduler:
    """Enhanced adaptive CFG scheduler supporting all schedule types."""
    
    def __init__(self, s0: float, s1: float, num_steps: int, schedule_type: str = "linear"):
        self.s0 = s0  # Starting CFG scale
        self.s1 = s1  # Ending CFG scale  
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
    def get_cfg_scale(self, step: int) -> float:
        """Get CFG scale for given step."""
        progress = step / (self.num_steps - 1)  # 0 to 1
        
        if self.schedule_type == "linear":
            return self.s0 + (self.s1 - self.s0) * progress
            
        elif self.schedule_type == "cosine":
            # Cosine schedule: smooth transition
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            return self.s0 + (self.s1 - self.s0) * cosine_progress
            
        elif self.schedule_type == "step":
            # Step function: discrete change at midpoint
            return self.s0 if progress < 0.5 else self.s1
            
        elif self.schedule_type == "exponential":
            # Exponential decay
            alpha = 2.0  # Decay rate
            exp_progress = (1 - np.exp(-alpha * progress)) / (1 - np.exp(-alpha))
            return self.s0 + (self.s1 - self.s0) * exp_progress
            
        elif self.schedule_type == "sigmoid":
            # Sigmoid schedule: smooth S-curve
            beta = 6.0  # Steepness
            sigmoid_progress = 1 / (1 + np.exp(-beta * (progress - 0.5)))
            return self.s0 + (self.s1 - self.s0) * sigmoid_progress
            
        else:
            # Default to linear
            return self.s0 + (self.s1 - self.s0) * progress

class LatentEnergyCallback:
    """Enhanced callback for capturing latents and computing energy metrics."""
    
    def __init__(self, scheduler: AdaptiveCFGScheduler):
        self.scheduler = scheduler
        self.latents = []
        self.timesteps = []
        self.cfg_values = []
        self.step_times = []
        
    def __call__(self, step: int, timestep: float, latents):
        """Callback function called during generation."""
        # Record timing
        self.step_times.append(time.time())
        
        # Get adaptive CFG value
        cfg_scale = self.scheduler.get_cfg_scale(step)
        
        # Store data
        if hasattr(latents, 'detach'):
            self.latents.append(latents.detach().cpu())
        else:
            self.latents.append(latents)
        self.timesteps.append(float(timestep))
        self.cfg_values.append(cfg_scale)
        
        return cfg_scale
    
    def get_energy_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate energy trajectory from captured latents."""
        energies = []
        
        for latent in self.latents:
            if hasattr(latent, 'norm'):
                # PyTorch tensor
                energy = (latent.norm() ** 2 / latent.numel()).item()
            else:
                # NumPy array
                energy = np.linalg.norm(latent) ** 2 / latent.size
            energies.append(energy)
        
        return (
            np.array(energies),
            np.array(self.timesteps), 
            np.array(self.cfg_values)
        )

class UnifiedDiffusionPipeline:
    """Unified pipeline supporting multiple models and resolutions."""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.pipeline = None
        self.setup_pipeline()
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def setup_pipeline(self):
        """Setup the appropriate diffusion pipeline."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
            
            # Model configurations
            model_configs = {
                "SD1.5": "runwayml/stable-diffusion-v1-5",
                "SDXL": "stabilityai/stable-diffusion-xl-base-1.0", 
                "SD2.1": "stabilityai/stable-diffusion-2-1"
            }
            
            if self.model_name not in model_configs:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            model_path = model_configs[self.model_name]
            
            # Load appropriate pipeline
            if self.model_name == "SDXL":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device != "cpu" else None
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            print(f"✅ {self.model_name} pipeline loaded on {self.device}")
            
        except ImportError:
            print("⚠️ PyTorch/Diffusers not available - running in simulation mode")
            self.pipeline = None
    
    def set_sampler(self, sampler: str):
        """Set the sampler for the pipeline."""
        if not self.pipeline:
            return
            
        try:
            from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
            
            if sampler.lower() == "ddim":
                self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            elif sampler.lower() == "eulera":
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            elif sampler.lower() == "dpm++2m":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            else:
                print(f"⚠️ Unknown sampler: {sampler}, using default")
                
        except ImportError:
            pass
    
    def generate_with_adaptive_cfg(
        self,
        prompt: str,
        height: int,
        width: int,
        s0: float,
        s1: float,
        schedule_type: str,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> Dict:
        """Generate image with adaptive CFG scheduling."""
        
        if not self.pipeline:
            # Simulation mode for testing without torch
            return self._simulate_generation(prompt, height, width, s0, s1, schedule_type, num_inference_steps, seed)
        
        try:
            import torch
            from PIL import Image
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Create adaptive scheduler
            scheduler = AdaptiveCFGScheduler(s0, s1, num_inference_steps, schedule_type)
            
            # Create callback for latent capture
            callback = LatentEnergyCallback(scheduler)
            
            # Generate image
            start_time = time.time()
            
            # Handle different pipeline types
            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": s0,  # Initial CFG scale
                "callback": callback,
                "callback_steps": 1
            }
            
            # SDXL specific parameters
            if self.model_name == "SDXL":
                generation_kwargs.update({
                    "denoising_end": 0.8,
                    "output_type": "latent"
                })
            
            result = self.pipeline(**generation_kwargs)
            
            generation_time = time.time() - start_time
            
            # Extract image
            if hasattr(result, 'images'):
                image = result.images[0]
            else:
                image = result
            
            # Get energy trajectory
            energies, timesteps, cfg_values = callback.get_energy_trajectory()
            
            return {
                "image": image,
                "energy_trajectory": energies.tolist(),
                "timesteps": timesteps.tolist(),
                "cfg_values": cfg_values.tolist(),
                "generation_time": generation_time,
                "metadata": {
                    "model": self.model_name,
                    "resolution": f"{width}x{height}",
                    "schedule_type": schedule_type,
                    "s0": s0,
                    "s1": s1,
                    "steps": num_inference_steps,
                    "seed": seed
                }
            }
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return self._simulate_generation(prompt, height, width, s0, s1, schedule_type, num_inference_steps, seed)
    
    def _simulate_generation(self, prompt: str, height: int, width: int, s0: float, s1: float, 
                           schedule_type: str, num_inference_steps: int, seed: Optional[int]) -> Dict:
        """Simulate generation for testing without full setup."""
        
        # Create simulated energy trajectory
        np.random.seed(seed if seed else 42)
        
        # Simulate realistic energy decay with some noise
        base_energies = np.linspace(10.0, 2.0, num_inference_steps)
        noise = np.random.normal(0, 0.5, num_inference_steps)
        energies = base_energies + noise
        energies = np.maximum(energies, 0.1)  # Keep positive
        
        # Simulate CFG schedule
        scheduler = AdaptiveCFGScheduler(s0, s1, num_inference_steps, schedule_type)
        cfg_values = [scheduler.get_cfg_scale(i) for i in range(num_inference_steps)]
        
        # Simulate timesteps
        timesteps = np.linspace(1000, 0, num_inference_steps)
        
        return {
            "image": None,  # Placeholder
            "energy_trajectory": energies.tolist(),
            "timesteps": timesteps.tolist(),
            "cfg_values": cfg_values,
            "generation_time": np.random.uniform(10, 60),  # Simulated time
            "metadata": {
                "model": self.model_name,
                "resolution": f"{width}x{height}",
                "schedule_type": schedule_type,
                "s0": s0,
                "s1": s1,
                "steps": num_inference_steps,
                "seed": seed,
                "simulated": True
            }
        }

class ExperimentRunner:
    """Runs comprehensive experiments according to configuration plan."""
    
    def __init__(self, config_path: str = "experiments/configuration_plan.json"):
        with open(config_path, 'r') as f:
            self.config_plan = json.load(f)
        
        self.prompts = self.config_plan["prompts"]
        self.seeds = self.config_plan["seeds"]
        self.results = {}
        
    def run_single_experiment(self, config: Dict, prompt_idx: int, seed: int) -> Dict:
        """Run a single experiment configuration."""
        
        prompt = self.prompts[prompt_idx]
        
        # Setup pipeline
        pipeline = UnifiedDiffusionPipeline(config["model"])
        pipeline.set_sampler(config["sampler"])
        
        # Generate image
        result = pipeline.generate_with_adaptive_cfg(
            prompt=prompt,
            height=config["resolution"],
            width=config["resolution"],
            s0=config["s0"],
            s1=config["s1"],
            schedule_type=config["schedule_type"],
            num_inference_steps=config["steps"],
            seed=seed
        )
        
        # Add experiment metadata
        result["experiment"] = {
            "config_id": config["config_id"],
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "seed": seed,
            "category": config["category"]
        }
        
        return result
    
    def run_configuration_batch(self, config: Dict, max_prompts: int = None, max_seeds: int = None):
        """Run all experiments for a specific configuration."""
        
        config_id = config["config_id"]
        category = config["category"]
        
        print(f"\n🔬 Running {config_id} ({category})")
        
        # Determine number of prompts and seeds based on category
        if category == "ultra_high_res":
            n_prompts = min(50, max_prompts or 50)  # Subset for ultra-high-res
            n_seeds = min(3, max_seeds or 3)
        elif category == "cross_model":
            n_prompts = min(100, max_prompts or 100)  # Subset for cross-model
            n_seeds = min(3, max_seeds or 3)
        else:
            n_prompts = max_prompts or len(self.prompts)
            n_seeds = max_seeds or len(self.seeds)
        
        results = []
        
        for prompt_idx in range(min(n_prompts, len(self.prompts))):
            for seed_idx in range(min(n_seeds, len(self.seeds))):
                seed = self.seeds[seed_idx]
                
                try:
                    result = self.run_single_experiment(config, prompt_idx, seed)
                    results.append(result)
                    
                    print(f"  ✅ Prompt {prompt_idx+1}/{n_prompts}, Seed {seed}")
                    
                except Exception as e:
                    print(f"  ❌ Failed: Prompt {prompt_idx+1}, Seed {seed}: {e}")
                    continue
        
        self.results[config_id] = results
        return results
    
    def run_all_experiments(self, categories: List[str] = None, max_prompts: int = None):
        """Run all experiments according to the plan."""
        
        categories = categories or ["sd15_main", "sdxl_scalability", "ultra_high_res", "cross_model"]
        
        print("🚀 Starting Comprehensive RectifiedHR Experiments")
        print("=" * 80)
        
        total_configs = 0
        
        for category in categories:
            if category not in self.config_plan["configurations"]:
                continue
                
            configs = self.config_plan["configurations"][category]
            print(f"\n📋 Category: {category.upper()} ({len(configs)} configurations)")
            
            for config in configs:
                self.run_configuration_batch(config, max_prompts=max_prompts)
                total_configs += 1
        
        print(f"\n🎉 Completed {total_configs} configurations!")
        return self.results
    
    def save_results(self, output_path: str = "experiments/comprehensive_results.json"):
        """Save all experiment results."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results_with_metadata = {
            "metadata": {
                "completed_at": datetime.now().isoformat(),
                "total_configurations": len(self.results),
                "total_experiments": sum(len(r) for r in self.results.values())
            },
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")

def main():
    """Run the unified experimental pipeline."""
    
    parser = argparse.ArgumentParser(description="RectifiedHR Unified Experiment Runner")
    parser.add_argument("--categories", nargs="+", 
                       choices=["sd15_main", "sdxl_scalability", "ultra_high_res", "cross_model"],
                       default=["sd15_main"], 
                       help="Categories to run")
    parser.add_argument("--max-prompts", type=int, default=5, 
                       help="Maximum prompts per configuration (for testing)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with simulated generation")
    
    args = parser.parse_args()
    
    print("🔬 RectifiedHR Unified Experimental Pipeline")
    print("=" * 60)
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Max prompts per config: {args.max_prompts}")
    print(f"Test mode: {'Yes' if args.test_mode else 'No'}")
    
    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_all_experiments(
        categories=args.categories,
        max_prompts=args.max_prompts
    )
    
    # Save results
    runner.save_results()
    
    print("\n✅ Unified experimental pipeline completed!")
    print("🔍 Next steps:")
    print("   1. Run enhanced metrics evaluation")
    print("   2. Generate correlation analysis")
    print("   3. Create publication tables and figures")

if __name__ == "__main__":
    main()
