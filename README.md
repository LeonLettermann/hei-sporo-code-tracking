# hei-sporo-code-tracking

**Code for image analysis to track *Plasmodium* sporozoites in 3D gel assays imaged by spinning disc microscopy.**

This repository is part of the [hei-data-code](https://github.com/LeonLettermann/hei-data-code) project and provides Python/JAX tools for analyzing sporozoite movement and calculating traction forces. It is associated with an upcoming scientific publication.

📦 **Test data** is available at the [associated dataset repository](https://doi.org/10.11588/DATA/4YBYXE).

---

## Repository Structure

```text
hei-sporo-code-tracking/
├── Example.ipynb                     # Example for applying the image analysis pipeline
├── ImageAnalysisCode/
│   ├── general.py                    # General functions (e.g., data loading)
│   ├── filter.py                     # Basic filtering functions
│   ├── shiftcorrect.py               # Drift correction via image shifting
│   ├── deconv.py                     # Blind deconvolution
│   ├── track.py                      # 4D image stack tracking of sporozoites
│   ├── trajanalysis.py               # Trajectory analysis
│   ├── plot.py                       # Plotting and visualization utilities
│   ├── pvexp.py                      # Export to ParaView-compatible formats
│   ├── pipelines.py                  # End-to-end analysis workflows
│   └── structanalysis.py             # (Experimental) Structure interaction analysis
├── TractionForce/
│   ├── AnalyzeTFSF.py                # Core code for two-sided traction force analysis
│   └── AnalysisPipeline.ipynb        # Example notebook for running traction force analysis
└── environment.yml                   # Conda environment file for image analysis
                                      # ⚠️ JAX version might need to be adapted to your system