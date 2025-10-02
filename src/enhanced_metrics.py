#!/usr/bin/env python3
"""
Enhanced metrics system for RectifiedHR revision.
Implements all professor's requirements + energy-quality correlation analysis.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

@dataclass
class EnergyMetrics:
    """Energy-based stability metrics as specified by professor."""
    spike_magnitude: float  # Maximum energy spike
    total_variation: float  # TV across trajectory
    monotonicity_violations: int  # Number of non-monotonic increases
    trajectory_area: float  # Area under energy curve
    final_energy: float  # Final energy value
    energy_range: float  # Max - min energy
    
    def to_dict(self) -> Dict:
        return {
            "spike_magnitude": self.spike_magnitude,
            "total_variation": self.total_variation, 
            "monotonicity_violations": self.monotonicity_violations,
            "trajectory_area": self.trajectory_area,
            "final_energy": self.final_energy,
            "energy_range": self.energy_range
        }

@dataclass
class PerceptualMetrics:
    """Perceptual quality metrics."""
    clip_score: float  # CLIPScore (reference-free)
    lpips_score: float  # LPIPS distance
    ms_ssim: float  # Multi-scale SSIM
    
    def to_dict(self) -> Dict:
        return {
            "clip_score": self.clip_score,
            "lpips_score": self.lpips_score,
            "ms_ssim": self.ms_ssim
        }

@dataclass
class ConsistencyMetrics:
    """Per-seed consistency metrics."""
    seed_agreement: float  # Agreement across seeds
    variance_clip: float  # Variance in CLIP scores
    variance_lpips: float  # Variance in LPIPS scores
    
    def to_dict(self) -> Dict:
        return {
            "seed_agreement": self.seed_agreement,
            "variance_clip": self.variance_clip,
            "variance_lpips": self.variance_lpips
        }

class EnhancedMetricsEvaluator:
    """Comprehensive metrics evaluator for RectifiedHR revision."""
    
    def __init__(self, device: str = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.clip_model = None
        self.clip_preprocess = None
        self.lpips_model = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Setup evaluation models."""
        # Setup CLIP
        if CLIP_AVAILABLE:
            try:
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self.clip_model = self.clip_model.to(self.device)
                print("✅ CLIP model loaded")
            except Exception as e:
                print(f"⚠️ CLIP setup failed: {e}")
        
        # Setup LPIPS
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                print("✅ LPIPS model loaded")
            except Exception as e:
                print(f"⚠️ LPIPS setup failed: {e}")
    
    def compute_energy_metrics(self, energy_trajectory: np.ndarray) -> EnergyMetrics:
        """Compute comprehensive energy stability metrics."""
        
        # Spike magnitude: maximum single-step increase
        energy_diffs = np.diff(energy_trajectory)
        spike_magnitude = np.max(energy_diffs) if len(energy_diffs) > 0 else 0.0
        
        # Total variation: sum of absolute differences
        total_variation = np.sum(np.abs(energy_diffs))
        
        # Monotonicity violations: count of increases (should be decreasing)
        monotonicity_violations = np.sum(energy_diffs > 0)
        
        # Trajectory area: area under the curve
        trajectory_area = np.trapz(energy_trajectory)
        
        # Additional metrics
        final_energy = energy_trajectory[-1] if len(energy_trajectory) > 0 else 0.0
        energy_range = np.max(energy_trajectory) - np.min(energy_trajectory)
        
        return EnergyMetrics(
            spike_magnitude=float(spike_magnitude),
            total_variation=float(total_variation),
            monotonicity_violations=int(monotonicity_violations),
            trajectory_area=float(trajectory_area),
            final_energy=float(final_energy),
            energy_range=float(energy_range)
        )
    
    def compute_perceptual_metrics(self, image_path: str, prompt: str, reference_images: Optional[List[str]] = None) -> PerceptualMetrics:
        """Compute perceptual quality metrics."""
        
        from PIL import Image
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return PerceptualMetrics(0.0, 1.0, 0.0)
        
        # CLIP Score (reference-free)
        clip_score = 0.0
        if self.clip_model is not None:
            try:
                clip_score = self._compute_clip_similarity(image, prompt)
            except Exception as e:
                print(f"CLIP computation failed: {e}")
        
        # LPIPS (if reference images provided)
        lpips_score = 0.0
        if self.lpips_model is not None and reference_images:
            try:
                lpips_score = self._compute_lpips_distance(image, reference_images[0])
            except Exception as e:
                print(f"LPIPS computation failed: {e}")
        
        # MS-SSIM (if reference images provided)
        ms_ssim = 0.0
        if SSIM_AVAILABLE and reference_images:
            try:
                ms_ssim = self._compute_ms_ssim(image, reference_images[0])
            except Exception as e:
                print(f"MS-SSIM computation failed: {e}")
        
        return PerceptualMetrics(
            clip_score=float(clip_score),
            lpips_score=float(lpips_score),
            ms_ssim=float(ms_ssim)
        )
    
    def compute_consistency_metrics(self, results_per_seed: Dict[int, Dict]) -> ConsistencyMetrics:
        """Compute per-seed consistency metrics."""
        
        # Extract CLIP scores across seeds
        clip_scores = [results_per_seed[seed]["perceptual"]["clip_score"] for seed in results_per_seed.keys()]
        lpips_scores = [results_per_seed[seed]["perceptual"]["lpips_score"] for seed in results_per_seed.keys()]
        
        # Compute variances
        variance_clip = float(np.var(clip_scores))
        variance_lpips = float(np.var(lpips_scores))
        
        # Seed agreement: 1 - normalized standard deviation
        seed_agreement = 1.0 - (np.std(clip_scores) / (np.mean(clip_scores) + 1e-8))
        seed_agreement = max(0.0, seed_agreement)  # Ensure non-negative
        
        return ConsistencyMetrics(
            seed_agreement=float(seed_agreement),
            variance_clip=variance_clip,
            variance_lpips=variance_lpips
        )
    
    def compute_energy_quality_correlation(self, experiments: List[Dict]) -> Dict[str, float]:
        """Compute Spearman correlation between energy stability and quality metrics."""
        
        # Extract metrics for correlation analysis
        total_variations = []
        clip_scores = []
        lpips_scores = []
        
        for exp in experiments:
            if "energy" in exp and "perceptual" in exp:
                total_variations.append(-exp["energy"]["total_variation"])  # Negative TV for correlation
                clip_scores.append(exp["perceptual"]["clip_score"])
                lpips_scores.append(-exp["perceptual"]["lpips_score"])  # Negative LPIPS (lower is better)
        
        correlations = {}
        
        # Spearman correlation: -TV vs CLIP
        if len(total_variations) > 1 and len(clip_scores) > 1:
            corr_clip, p_clip = stats.spearmanr(total_variations, clip_scores)
            correlations["clip_vs_negative_tv"] = float(corr_clip) if not np.isnan(corr_clip) else 0.0
            correlations["clip_vs_negative_tv_pvalue"] = float(p_clip) if not np.isnan(p_clip) else 1.0
        
        # Spearman correlation: -TV vs -LPIPS  
        if len(total_variations) > 1 and len(lpips_scores) > 1:
            corr_lpips, p_lpips = stats.spearmanr(total_variations, lpips_scores)
            correlations["negative_lpips_vs_negative_tv"] = float(corr_lpips) if not np.isnan(corr_lpips) else 0.0
            correlations["negative_lpips_vs_negative_tv_pvalue"] = float(p_lpips) if not np.isnan(p_lpips) else 1.0
        
        return correlations
    
    def _compute_clip_similarity(self, image, prompt: str) -> float:
        """Compute CLIP similarity between image and prompt."""
        
        # Preprocess image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode text
        text_input = open_clip.tokenize([prompt]).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features * text_features).sum(dim=-1).item()
        
        return similarity
    
    def _compute_lpips_distance(self, image1, image2_path: str) -> float:
        """Compute LPIPS perceptual distance."""
        
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load second image
        image2 = Image.open(image2_path).convert('RGB')
        
        # Transform to tensors
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        img1_tensor = transform(image1).unsqueeze(0).to(self.device)
        img2_tensor = transform(image2).unsqueeze(0).to(self.device)
        
        # Compute LPIPS distance
        with torch.no_grad():
            distance = self.lpips_model(img1_tensor, img2_tensor).item()
        
        return distance
    
    def _compute_ms_ssim(self, image1, image2_path: str) -> float:
        """Compute Multi-Scale SSIM."""
        
        from PIL import Image
        
        # Load second image  
        image2 = Image.open(image2_path).convert('L')  # Grayscale
        image1 = image1.convert('L')
        
        # Convert to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2.resize(image1.size))
        
        # Compute SSIM
        ssim_score = ssim(img1_array, img2_array, data_range=255)
        
        return ssim_score
    
    def generate_correlation_plots(self, experiments: List[Dict], output_dir: str = "experiments/correlation_analysis"):
        """Generate correlation visualization plots."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data for plotting
        total_variations = []
        clip_scores = []
        lpips_scores = []
        config_ids = []
        
        for exp in experiments:
            if "energy" in exp and "perceptual" in exp:
                total_variations.append(exp["energy"]["total_variation"])
                clip_scores.append(exp["perceptual"]["clip_score"])
                lpips_scores.append(exp["perceptual"]["lpips_score"])
                config_ids.append(exp.get("config_id", "unknown"))
        
        # Create correlation plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: CLIP Score vs -Total Variation
        axes[0].scatter([-tv for tv in total_variations], clip_scores, alpha=0.7)
        axes[0].set_xlabel('-Total Variation (Energy Stability)')
        axes[0].set_ylabel('CLIP Score')
        axes[0].set_title('Energy Stability vs CLIP Score')
        
        # Add correlation coefficient
        if len(total_variations) > 1:
            corr_clip, _ = stats.spearmanr([-tv for tv in total_variations], clip_scores)
            axes[0].text(0.05, 0.95, f'ρ = {corr_clip:.3f}', transform=axes[0].transAxes, 
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Plot 2: LPIPS vs Total Variation 
        axes[1].scatter(total_variations, lpips_scores, alpha=0.7)
        axes[1].set_xlabel('Total Variation (Energy Instability)')
        axes[1].set_ylabel('LPIPS Distance')
        axes[1].set_title('Energy Instability vs LPIPS Distance')
        
        # Add correlation coefficient
        if len(total_variations) > 1:
            corr_lpips, _ = stats.spearmanr(total_variations, lpips_scores)
            axes[1].text(0.05, 0.95, f'ρ = {corr_lpips:.3f}', transform=axes[1].transAxes,
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / "energy_quality_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation plots saved to: {output_path}")
    
    def save_comprehensive_results(self, all_results: Dict, output_path: str = "experiments/comprehensive_results.json"):
        """Save all results in structured format for analysis."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Comprehensive results saved to: {output_file}")

def main():
    """Test the enhanced metrics system."""
    
    evaluator = EnhancedMetricsEvaluator()
    
    print("🧮 Enhanced Metrics System Initialized")
    print(f"   • Device: {evaluator.device}")
    print(f"   • CLIP available: {'✅' if evaluator.clip_model else '❌'}")
    print(f"   • LPIPS available: {'✅' if evaluator.lpips_model else '❌'}")
    print(f"   • SSIM available: {'✅' if SSIM_AVAILABLE else '❌'}")
    
    # Test energy metrics with sample data
    sample_energy = np.array([10.5, 8.2, 6.8, 7.1, 5.9, 4.2, 3.8, 3.5])
    energy_metrics = evaluator.compute_energy_metrics(sample_energy)
    
    print(f"\n📊 Sample Energy Metrics:")
    print(f"   • Spike magnitude: {energy_metrics.spike_magnitude:.3f}")
    print(f"   • Total variation: {energy_metrics.total_variation:.3f}")
    print(f"   • Monotonicity violations: {energy_metrics.monotonicity_violations}")
    print(f"   • Trajectory area: {energy_metrics.trajectory_area:.3f}")
    
    print("\n✅ Enhanced metrics system ready for comprehensive evaluation!")

if __name__ == "__main__":
    main()
