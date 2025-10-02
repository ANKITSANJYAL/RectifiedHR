# Generated Images for RectifiedHR Paper

## 📁 Image Directory Structure

All generated images are automatically saved to organized directories:

### `experiments/baseline/` (219 images)
**Purpose**: Standard diffusion model results for comparison
**Format**: `baseline_{sampler}_cfg{scale}_{resolution}_prompt{N}_{timestamp}.png`
**Examples**:
- `baseline_ddim_cfg10.0_512_prompt1_20250712_092300.png`
- `baseline_euler_a_cfg7.0_512_prompt5_20250712_095202.png`
- `baseline_dpm++_2m_cfg15.0_512_prompt3_20250712_100258.png`

### `experiments/adaptive_cfg/` (40 images)
**Purpose**: RectifiedHR adaptive CFG results
**Format**: `adaptive_{schedule_type}_cfg{scale}_{resolution}_prompt{N}_{timestamp}.png`
**Schedule Types**:
- `cosine_ramp` - Smooth cosine increase
- `cosine_inverse` - Smooth cosine decrease  
- `linear_increasing` - Linear ramp up
- `linear_decreasing` - Linear ramp down
- `step_function` - Step-wise changes

**Examples**:
- `adaptive_cosine_ramp_cfg7.0_512_prompt1_20250712_120919.png`
- `adaptive_linear_decreasing_cfg7.0_512_prompt4_20250712_121646.png`

### `experiments/energy_plots/` (Research Figures)
**Purpose**: Publication-ready figures for paper
**Contents**:
- `figure1_energy_evolution.png` - Energy trajectory analysis
- `figure2_energy_stability.png` - Stability comparison
- `figure3_final_energy_correlation.png` - Final energy correlation
- `comprehensive_energy_analysis.png` - Complete analysis

### `paper/figures/` (Paper Assets)
**Purpose**: Curated figures for LaTeX document
**Contents**: Selected best images for paper inclusion

## 📊 Image Metadata

Each image has associated metadata stored as JSON:
- **Baseline**: `metadata_{timestamp}.json` 
- **Adaptive**: `latents_adaptive_{schedule}_{timestamp}.json`
- **Energy**: Embedded in energy analysis files

## 🎯 Key Images for Paper

### Main Results Comparison:
1. **Baseline vs Adaptive** - Same prompt, different methods
2. **Energy Evolution** - Shows CFG schedule effectiveness  
3. **Quality Metrics** - CLIP, LPIPS, MS-SSIM comparisons
4. **Artifact Reduction** - High CFG vs stabilized output

### Recommended Paper Figures:
- `experiments/energy_plots/figure1_energy_evolution.png`
- `experiments/energy_plots/figure2_energy_stability.png` 
- Best adaptive vs baseline pairs for visual comparison
- Metrics comparison charts (generated during evaluation)

## 💡 Usage Notes

- All images are 512×512 for memory efficiency
- Timestamps ensure unique filenames
- JSON files contain complete generation parameters
- Energy profiles enable detailed analysis
- Ready for direct inclusion in research paper
