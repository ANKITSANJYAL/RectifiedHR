#!/usr/bin/env python3
"""
Baseline image generation for RectifiedHR research.
Generates high-resolution images using Stable Diffusion and saves intermediate latents.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import json
from datetime import datetime

class LatentCallback:
    """Callback to capture intermediate latents during generation."""
    
    def __init__(self):
        self.latents = []
        self.timesteps = []
        
    def __call__(self, i, t, latents):
        self.latents.append(latents.detach().cpu())
        self.timesteps.append(t.item())
        
    def get_energy_profile(self):
        """Calculate energy profile from captured latents."""
        energies = []
        for latent in self.latents:
            # Energy calculation: E_t = ||latent||^2 / num_elements
            energy = torch.norm(latent) ** 2 / latent.numel()
            energies.append(energy.item())
        return np.array(energies), np.array(self.timesteps)

def setup_pipeline(model_name="runwayml/stable-diffusion-v1-5", sampler="ddim"):
    """Setup the diffusion pipeline with specified sampler."""
    
    # Load pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Setup scheduler
    if sampler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif sampler == "euler_a":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif sampler == "dpm++_2m":
        from diffusers import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    # Move to device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    return pipeline

def generate_baseline_images(
    prompt,
    cfg_scale=7.0,
    sampler="ddim",
    resolution=1024,
    num_inference_steps=50,
    model_name="runwayml/stable-diffusion-v1-5",
    output_dir="experiments/baseline",
    save_latents=True,
    suffix=""
):
    """Generate baseline images and save intermediate latents."""
    
    # Setup pipeline
    pipeline = setup_pipeline(model_name, sampler)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup callback for latent capture
    callback = LatentCallback()
    
    # Generate image
    print(f"Generating image with prompt: '{prompt}'")
    print(f"CFG Scale: {cfg_scale}, Sampler: {sampler}, Resolution: {resolution}")
    
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            callback=callback,
            callback_steps=1,
            return_dict=True
        )
    
    # Save generated image
    image = result.images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_part = f"_{suffix}" if suffix else ""
    image_filename = f"baseline_{sampler}_cfg{cfg_scale}_{resolution}{suffix_part}_{timestamp}.png"
    image_path = output_path / image_filename
    image.save(image_path)
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "cfg_scale": cfg_scale,
        "sampler": sampler,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "model_name": model_name,
        "image_filename": image_filename,
        "timestamp": timestamp
    }
    
    metadata_path = output_path / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save latent energy profile
    if save_latents:
        energies, timesteps = callback.get_energy_profile()
        
        latent_data = {
            "energies": energies.tolist(),
            "timesteps": timesteps.tolist(),
            "metadata": metadata
        }
        
        latent_filename = f"latents_{sampler}_cfg{cfg_scale}_{resolution}{suffix_part}_{timestamp}.json"
        latent_path = output_path / latent_filename
        with open(latent_path, 'w') as f:
            json.dump(latent_data, f, indent=2)
        
        print(f"Energy profile saved: {latent_path}")
        print(f"Final energy: {energies[-1]:.4f}")
    
    print(f"Image saved: {image_path}")
    return image_path, metadata

def generate_comprehensive_baseline(
    prompts=None,
    cfg_scales=None,
    samplers=None,
    resolutions=None,
    output_dir="experiments/baseline"
):
    """Generate comprehensive baseline images for research paper."""
    
    if prompts is None:
        prompts = [
            "a beautiful landscape with mountains and lake, high resolution, detailed",
            "a portrait of a woman with detailed facial features, professional photography",
            "a futuristic cityscape with skyscrapers and flying cars, cinematic lighting",
            "a close-up of a flower with intricate details, macro photography",
            "a medieval castle on a hill, dramatic lighting, fantasy art style"
        ]
    
    if cfg_scales is None:
        cfg_scales = [3, 5, 7, 10]
    
    if samplers is None:
        samplers = ["ddim", "euler_a"]
    
    if resolutions is None:
        resolutions = [512, 768]  # Avoid 1024 for memory issues
    
    print("Generating comprehensive baseline images for research paper...")
    print(f"Prompts: {len(prompts)}")
    print(f"CFG Scales: {cfg_scales}")
    print(f"Samplers: {samplers}")
    print(f"Resolutions: {resolutions}")
    
    results = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n--- Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt[:50]}...' ---")
        
        for resolution in resolutions:
            print(f"  Resolution: {resolution}")
            
            for sampler in samplers:
                print(f"    Sampler: {sampler}")
                
                for cfg_scale in cfg_scales:
                    print(f"      CFG Scale: {cfg_scale}")
                    
                    try:
                        image_path, metadata = generate_baseline_images(
                            prompt=prompt,
                            cfg_scale=cfg_scale,
                            sampler=sampler,
                            resolution=resolution,
                            output_dir=output_dir
                        )
                        results.append({
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "resolution": resolution,
                            "sampler": sampler,
                            "cfg_scale": cfg_scale,
                            "image_path": str(image_path),
                            "metadata": metadata
                        })
                    except Exception as e:
                        print(f"      Error: {e}")
                        continue
    
    # Save comprehensive results summary
    summary_path = Path(output_dir) / "comprehensive_baseline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive baseline generation complete!")
    print(f"Generated {len(results)} images")
    print(f"Summary saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate baseline images for RectifiedHR research")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and lake, high resolution, detailed", 
                       help="Text prompt for image generation")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "euler_a", "dpm++_2m"], 
                       help="Sampling method")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", 
                       help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="experiments/baseline", 
                       help="Output directory")
    parser.add_argument("--suffix", type=str, default="", 
                       help="Suffix for filename")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Generate comprehensive baseline for research paper")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        # Generate comprehensive baseline for research
        generate_comprehensive_baseline()
    else:
        # Generate single baseline image
        image_path, metadata = generate_baseline_images(
            prompt=args.prompt,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            resolution=args.resolution,
            num_inference_steps=args.num_steps,
            model_name=args.model,
            output_dir=args.output_dir,
            suffix=args.suffix
        )
        
        print(f"\nGeneration complete!")
        print(f"Image: {image_path}")
        print(f"Metadata: experiments/baseline/metadata_{metadata['timestamp']}.json")

if __name__ == "__main__":
    main() 