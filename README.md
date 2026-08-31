The interface batch analysis and plot interface results are designed to analyze the interactions between a protein binder and target. They can be used by changing the selected hotspot residues of interest in the script and using the .pdb files of the (predicted) protein-protein complex.

This repository provides two core scripts—interface_batch_analysis and plot_interface_results—designed to analyze and visualize the molecular interactions between a protein binder and its target.🚀 OverviewThese tools automate the evaluation of protein-protein complexes using structural data.interface_batch_analysis: Extracts interaction data from a batch of structures based on user-defined hotspot residues.plot_interface_results: Generates visual plots of the interaction profiles to help identify key binding trends.📋 Prerequisites & Input DataTo use these scripts, you will need:.pdb files: Structural files of your (predicted) protein-protein complexes (e.g., from AlphaFold-Multimer, ESMFold, or experimental data).Python 3.x with required dependencies (e.g., biopython, matplotlib, pandas, seaborn).

How to Use:
1. Configure Hotspot ResiduesOpen the analysis script and modify the target hotspot residues of interest to match your specific protein system:python# Edit this section in the script
HOTSPOT_RESIDUES = [32, 34, 38, 42]  # Example residue numbers
TARGET_CHAIN = "A"
BINDER_CHAIN = "B"
2. Run the Batch AnalysisPlace your .pdb files in the designated input directory and run the batch script to extract binding interface data:bashpython interface_batch_analysis.py --input_dir ./pdbs --output results.csv
3. Plot the ResultsVisualize the interaction data generated in the previous step:bashpython plot_interface_results.py --input results.csv --output_dir ./plots

Installation

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd Interface-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
