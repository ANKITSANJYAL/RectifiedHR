#!/usr/bin/env python3
"""
Evaluation metrics for RectifiedHR research.
Computes CLIP similarity, MS-SSIM, and other metrics for comparing image quality.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import glob

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install open_clip_torch")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

class ImageEvaluator:
    """Evaluates image quality using various metrics."""
    
    def __init__(self, device: str = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_preprocess = None
        
        if CLIP_AVAILABLE:
            self._setup_clip()
    
    def _setup_clip(self):
        """Setup CLIP model for evaluation."""
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.clip_model = self.clip_model.to(self.device)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.clip_model = None
    
    def compute_clip_similarity(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity between image and text prompt."""
        if not self.clip_model:
            return 0.0
        
        try:
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
        except Exception as e:
            print(f"Error computing CLIP similarity: {e}")
            return 0.0
    
    def compute_ms_ssim(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compute MS-SSIM between two images."""
        if not SSIM_AVAILABLE:
            return 0.0
        
        try:
            # Convert to numpy arrays
            img1 = np.array(image1.convert('L'))  # Convert to grayscale
            img2 = np.array(image2.convert('L'))
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = np.array(Image.fromarray(img2).resize(img1.shape[::-1]))
            
            # Compute SSIM
            ssim_score = ssim(img1, img2, data_range=255)
            return ssim_score
        except Exception as e:
            print(f"Error computing MS-SSIM: {e}")
            return 0.0
    
    def compute_lpips(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compute LPIPS distance between two images."""
        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex').to(self.device)
            
            # Convert to tensors
            def pil_to_tensor(pil_img):
                img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
                return img.unsqueeze(0).to(self.device)
            
            img1_tensor = pil_to_tensor(image1)
            img2_tensor = pil_to_tensor(image2)
            
            # Compute LPIPS
            with torch.no_grad():
                lpips_score = loss_fn(img1_tensor, img2_tensor).item()
            
            return lpips_score
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            return 0.0
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            return 0.0
    
    def compute_psnr(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compute PSNR between two images."""
        try:
            # Convert to numpy arrays
            img1 = np.array(image1).astype(np.float32)
            img2 = np.array(image2).astype(np.float32)
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = np.array(Image.fromarray(img2.astype(np.uint8)).resize(img1.shape[1::-1])).astype(np.float32)
            
            # Compute MSE
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            
            # Compute PSNR
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return psnr
        except Exception as e:
            print(f"Error computing PSNR: {e}")
            return 0.0
    
    def evaluate_single_image(self, image_path: str, prompt: str, reference_image: str = None) -> Dict:
        """Evaluate a single image with all available metrics."""
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {}
        
        results = {
            "image_path": image_path,
            "prompt": prompt
        }
        
        # CLIP similarity
        if self.clip_model:
            clip_score = self.compute_clip_similarity(image, prompt)
            results["clip_similarity"] = clip_score
        
        # Compare with reference image if provided
        if reference_image:
            try:
                ref_image = Image.open(reference_image)
                
                # MS-SSIM
                if SSIM_AVAILABLE:
                    ms_ssim_score = self.compute_ms_ssim(image, ref_image)
                    results["ms_ssim"] = ms_ssim_score
                
                # LPIPS
                lpips_score = self.compute_lpips(image, ref_image)
                results["lpips"] = lpips_score
                
                # PSNR
                psnr_score = self.compute_psnr(image, ref_image)
                results["psnr"] = psnr_score
                
            except Exception as e:
                print(f"Error comparing with reference image: {e}")
        
        return results

def load_evaluation_data(data_dir: str) -> Dict:
    """Load evaluation data from experiment directories."""
    data = {}
    
    # Load baseline data
    baseline_dir = Path(data_dir) / "baseline"
    if baseline_dir.exists():
        data["baseline"] = {}
        for metadata_file in baseline_dir.glob("metadata_*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                image_file = baseline_dir / metadata["image_filename"]
                if image_file.exists():
                    data["baseline"][metadata["timestamp"]] = {
                        "image_path": str(image_file),
                        "metadata": metadata
                    }
    
    # Load adaptive CFG data
    adaptive_dir = Path(data_dir) / "adaptive_cfg"
    if adaptive_dir.exists():
        data["adaptive_cfg"] = {}
        for metadata_file in adaptive_dir.glob("metadata_*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                image_file = adaptive_dir / metadata["image_filename"]
                if image_file.exists():
                    data["adaptive_cfg"][metadata["timestamp"]] = {
                        "image_path": str(image_file),
                        "metadata": metadata
                    }
    
    return data

def compare_methods(evaluator: ImageEvaluator, data: Dict, output_dir: str = "experiments/comparisons"):
    """Compare different methods using evaluation metrics."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Evaluate baseline images
    if "baseline" in data:
        print("Evaluating baseline images...")
        for timestamp, item in data["baseline"].items():
            prompt = item["metadata"]["prompt"]
            image_path = item["image_path"]
            
            result = evaluator.evaluate_single_image(image_path, prompt)
            result["method"] = "baseline"
            result["metadata"] = item["metadata"]
            results[f"baseline_{timestamp}"] = result
    
    # Evaluate adaptive CFG images
    if "adaptive_cfg" in data:
        print("Evaluating adaptive CFG images...")
        for timestamp, item in data["adaptive_cfg"].items():
            prompt = item["metadata"]["prompt"]
            image_path = item["image_path"]
            
            result = evaluator.evaluate_single_image(image_path, prompt)
            result["method"] = "adaptive_cfg"
            result["metadata"] = item["metadata"]
            results[f"adaptive_cfg_{timestamp}"] = result
    
    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary
    summary = generate_evaluation_summary(results)
    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {results_file}")
    print(f"Evaluation summary saved to: {summary_file}")
    
    return results, summary

def generate_evaluation_summary(results: Dict) -> Dict:
    """Generate summary statistics from evaluation results."""
    
    summary = {
        "total_images": len(results),
        "methods": {},
        "metrics_summary": {}
    }
    
    # Group by method
    methods = {}
    for key, result in results.items():
        method = result.get("method", "unknown")
        if method not in methods:
            methods[method] = []
        methods[method].append(result)
    
    # Calculate statistics for each method
    for method, method_results in methods.items():
        summary["methods"][method] = {
            "count": len(method_results),
            "metrics": {}
        }
        
        # Calculate metric statistics
        metrics = ["clip_similarity", "ms_ssim", "lpips", "psnr"]
        for metric in metrics:
            values = [r.get(metric, 0) for r in method_results if r.get(metric) is not None]
            if values:
                summary["methods"][method]["metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
    
    # Overall metric summary
    for metric in metrics:
        all_values = []
        for result in results.values():
            if result.get(metric) is not None:
                all_values.append(result[metric])
        
        if all_values:
            summary["metrics_summary"][metric] = {
                "mean": np.mean(all_values),
                "std": np.std(all_values),
                "min": np.min(all_values),
                "max": np.max(all_values)
            }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate image quality metrics for RectifiedHR research")
    parser.add_argument("--data_dir", type=str, default="experiments", 
                       help="Directory containing experiment data")
    parser.add_argument("--output_dir", type=str, default="experiments/comparisons", 
                       help="Output directory for evaluation results")
    parser.add_argument("--image_path", type=str, 
                       help="Evaluate single image")
    parser.add_argument("--prompt", type=str, 
                       help="Text prompt for single image evaluation")
    parser.add_argument("--reference", type=str, 
                       help="Reference image for comparison")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ImageEvaluator()
    
    if args.image_path:
        # Evaluate single image
        if not args.prompt:
            print("Error: --prompt is required for single image evaluation")
            return
        
        result = evaluator.evaluate_single_image(args.image_path, args.prompt, args.reference)
        print("Single image evaluation results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        # Evaluate all experiments
        print(f"Loading evaluation data from: {args.data_dir}")
        data = load_evaluation_data(args.data_dir)
        
        if not data:
            print("No evaluation data found. Please run generate.py and adaptive_cfg.py first.")
            return
        
        print(f"Found {sum(len(method_data) for method_data in data.values())} images to evaluate")
        
        # Compare methods
        results, summary = compare_methods(evaluator, data, args.output_dir)
        
        # Print summary
        print("\nEvaluation Summary:")
        for method, method_summary in summary["methods"].items():
            print(f"\n{method.upper()}:")
            print(f"  Images: {method_summary['count']}")
            for metric, stats in method_summary["metrics"].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

if __name__ == "__main__":
    main() 