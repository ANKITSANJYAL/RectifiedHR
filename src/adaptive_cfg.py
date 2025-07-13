#!/usr/bin/env python3
"""
Adaptive CFG scheduling for RectifiedHR research.
Implements rectified guidance scheduling to improve high-resolution image quality.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import json
from datetime import datetime
from typing import Callable, List

class AdaptiveCFGScheduler:
    """Implements adaptive CFG scheduling strategies."""
    
    def __init__(self, base_cfg: float = 7.0, num_steps: int = 50):
        self.base_cfg = base_cfg
        self.num_steps = num_steps
        
    def linear_increasing(self, step: int) -> float:
        """Linear increasing CFG schedule."""
        progress = step / self.num_steps
        return self.base_cfg * (0.5 + 0.5 * progress)
    
    def linear_decreasing(self, step: int) -> float:
        """Linear decreasing CFG schedule."""
        progress = step / self.num_steps
        return self.base_cfg * (1.5 - 0.5 * progress)
    
    def cosine_ramp(self, step: int) -> float:
        """Cosine ramp CFG schedule."""
        progress = step / self.num_steps
        return self.base_cfg * (0.5 + 0.5 * np.cos(np.pi * (1 - progress)))
    
    def cosine_inverse(self, step: int) -> float:
        """Inverse cosine CFG schedule."""
        progress = step / self.num_steps
        return self.base_cfg * (1.5 - 0.5 * np.cos(np.pi * progress))
    
    def step_function(self, step: int) -> float:
        """Step function CFG schedule."""
        if step < self.num_steps // 3:
            return self.base_cfg * 0.5
        elif step < 2 * self.num_steps // 3:
            return self.base_cfg * 1.0
        else:
            return self.base_cfg * 1.5
    
    def get_schedule(self, schedule_type: str) -> Callable:
        """Get CFG schedule function."""
        schedules = {
            'linear_increasing': self.linear_increasing,
            'linear_decreasing': self.linear_decreasing,
            'cosine_ramp': self.cosine_ramp,
            'cosine_inverse': self.cosine_inverse,
            'step_function': self.step_function
        }
        return schedules.get(schedule_type, self.linear_increasing)

class AdaptiveCFGCallback:
    """Callback for adaptive CFG scheduling with latent capture."""
    
    def __init__(self, scheduler: AdaptiveCFGScheduler, schedule_type: str):
        self.scheduler = scheduler
        self.schedule_func = scheduler.get_schedule(schedule_type)
        self.latents = []
        self.timesteps = []
        self.cfg_values = []
        
    def __call__(self, i: int, t: torch.Tensor, latents: torch.Tensor):
        # Calculate adaptive CFG value
        cfg_value = self.schedule_func(i)
        
        # Store data
        self.latents.append(latents.detach().cpu())
        self.timesteps.append(t.item())
        self.cfg_values.append(cfg_value)
        
        return cfg_value
    
    def get_energy_profile(self):
        """Calculate energy profile from captured latents."""
        energies = []
        for latent in self.latents:
            energy = torch.norm(latent) ** 2 / latent.numel()
            energies.append(energy.item())
        return np.array(energies), np.array(self.timesteps), np.array(self.cfg_values)

def setup_adaptive_pipeline(model_name: str = "runwayml/stable-diffusion-v1-5"):
    """Setup pipeline for adaptive CFG generation."""
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DDIM scheduler for better control
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    return pipeline

def generate_with_adaptive_cfg(
    pipeline,
    prompt: str,
    scheduler: AdaptiveCFGScheduler,
    schedule_type: str = "cosine_ramp",
    resolution: int = 1024,
    num_inference_steps: int = 50,
    output_dir: str = "experiments/adaptive_cfg",
    suffix: str = ""
):
    """Generate image with adaptive CFG scheduling."""
    
    # Setup callback
    callback = AdaptiveCFGCallback(scheduler, schedule_type)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating with adaptive CFG: {schedule_type}")
    print(f"Base CFG: {scheduler.base_cfg}, Steps: {num_inference_steps}")
    
    # Custom generation loop for adaptive CFG
    with torch.no_grad():
        # Prepare inputs
        device = pipeline.device
        dtype = pipeline.unet.dtype
        
        # Encode prompt
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Get text embeddings
        text_embeddings = pipeline.text_encoder(text_input_ids)[0]
        
        # Create uncond embeddings
        max_length = text_input_ids.shape[-1]
        uncond_input_ids = pipeline.tokenizer(
            [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
        ).input_ids.to(device)
        uncond_embeddings = pipeline.text_encoder(uncond_input_ids)[0]
        
        # Concatenate embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latents
        latents = torch.randn(
            (1, pipeline.unet.config.in_channels, resolution // 8, resolution // 8),
            device=device,
            dtype=dtype
        )
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        # Denoising loop
        pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = pipeline.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            # Get adaptive CFG value
            cfg_value = callback(i, t, latents)
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_value * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_part = f"_{suffix}" if suffix else ""
    image_filename = f"adaptive_{schedule_type}_cfg{scheduler.base_cfg}_{resolution}{suffix_part}_{timestamp}.png"
    image_path = output_path / image_filename
    image.save(image_path)
    
    # Save metadata and energy profile
    metadata = {
        "prompt": prompt,
        "schedule_type": schedule_type,
        "base_cfg": scheduler.base_cfg,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "image_filename": image_filename,
        "timestamp": timestamp
    }
    
    # Save energy profile
    energies, timesteps, cfg_values = callback.get_energy_profile()
    
    latent_data = {
        "energies": energies.tolist(),
        "timesteps": timesteps.tolist(),
        "cfg_values": cfg_values.tolist(),
        "metadata": metadata
    }
    
    latent_filename = f"latents_adaptive_{schedule_type}_cfg{scheduler.base_cfg}_{resolution}{suffix_part}_{timestamp}.json"
    latent_path = output_path / latent_filename
    with open(latent_path, 'w') as f:
        json.dump(latent_data, f, indent=2)
    
    print(f"Image saved: {image_path}")
    print(f"Energy profile saved: {latent_path}")
    print(f"Final energy: {energies[-1]:.4f}")
    
    return image_path, metadata, latent_data

def compare_schedules(
    prompt: str,
    base_cfg: float = 7.0,
    resolution: int = 1024,
    num_steps: int = 50,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "experiments/adaptive_cfg",
    suffix: str = ""
):
    """Compare different adaptive CFG schedules."""
    
    pipeline = setup_adaptive_pipeline(model_name)
    scheduler = AdaptiveCFGScheduler(base_cfg, num_steps)
    
    schedules = ["linear_increasing", "linear_decreasing", "cosine_ramp", "cosine_inverse", "step_function"]
    
    results = {}
    
    for schedule_type in schedules:
        print(f"\nTesting schedule: {schedule_type}")
        try:
            image_path, metadata, latent_data = generate_with_adaptive_cfg(
                pipeline, prompt, scheduler, schedule_type, resolution, num_steps, output_dir, suffix
            )
            results[schedule_type] = {
                "image_path": image_path,
                "metadata": metadata,
                "latent_data": latent_data
            }
        except Exception as e:
            print(f"Error with schedule {schedule_type}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Adaptive CFG scheduling for RectifiedHR research")
    parser.add_argument("--prompt", type=str, 
                       default="a beautiful landscape with mountains and lake, high resolution, detailed", 
                       help="Text prompt for image generation")
    parser.add_argument("--schedule", type=str, default="cosine_ramp", 
                       choices=["linear_increasing", "linear_decreasing", "cosine_ramp", "cosine_inverse", "step_function"],
                       help="CFG schedule type")
    parser.add_argument("--base_cfg", type=float, default=7.0, help="Base CFG scale")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", 
                       help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="experiments/adaptive_cfg", 
                       help="Output directory")
    parser.add_argument("--suffix", type=str, default="", 
                       help="Suffix for filename")
    parser.add_argument("--compare_all", action="store_true", 
                       help="Compare all schedule types")
    
    args = parser.parse_args()
    
    if args.compare_all:
        print("Comparing all adaptive CFG schedules...")
        results = compare_schedules(
            args.prompt, args.base_cfg, args.resolution, args.num_steps, 
            args.model, args.output_dir, args.suffix
        )
        print(f"\nComparison complete! Generated {len(results)} schedules.")
    else:
        # Generate single adaptive CFG image
        pipeline = setup_adaptive_pipeline(args.model)
        scheduler = AdaptiveCFGScheduler(args.base_cfg, args.num_steps)
        
        image_path, metadata, latent_data = generate_with_adaptive_cfg(
            pipeline, args.prompt, scheduler, args.schedule, 
            args.resolution, args.num_steps, args.output_dir, args.suffix
        )
        
        print(f"\nAdaptive CFG generation complete!")
        print(f"Image: {image_path}")

if __name__ == "__main__":
    main() 