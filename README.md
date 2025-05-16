# hei-sporo-code-tracking
Code for image analysis to track sporozoites in 3D gel assays imaged by spinning disc microscopy.

This repository is part of the hei-data-code (github.com/LeonLettermann/hei-data-code), and contains Python/JAX code for image analysis and traction force measurements of Plasmodium sporozoite movement, related to an upcomming publication. Data files to test the code provided here can be found in the data repository https://doi.org/10.11588/DATA/4YBYXE .

This repository contains
|____Example.ipynb              --->  Example for applying the image analysis code
|____ImageAnalysisCode          --->  The Code base for image analysis and sporozoite tracking
| |____general.py                   ---> General functions like data loading
| |____filter.py                    ---> Basic filter functions
| |____shiftcorrect.py              ---> Drift correcton by shifting images
| |____deconv.py                    ---> Blind deconvolution
| |____track.py                     ---> Tracking sporozoite in 4D image stacks
| |____trajanalysis.py              ---> Analyzing the trajectories resulting form tracking
| |____plot.py                      ---> Plot and visualization code
| |____pvexp.py                     ---> Export to paraview compatible files
| |____pipelines.py                 ---> Analysis workflows
| |____structanalysis.py            ---> Experimental, analyzing interactions with structures
|____TractionForce              --->  Code for the traction force analysis for two-sided traction force
| |____AnalyzeTFSF.py               ---> Code base
| |____AnalysisPipeline.ipynb       ---> Example for applying the code to example data