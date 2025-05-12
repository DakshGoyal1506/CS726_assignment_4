# CS726 Assignment 4

This repository contains code and files for Assignment 4 of CS726. Below is a brief overview:

## Structure

- **TASK-0-1 Folder**  
  - `sampling_algos.py`  
    - Implements two sampling algorithms (Algo1_Sampler and Algo2_Sampler).  
    - Main script runs both samplers on a trained energy model and visualizes results with t-SNE.
  - `plotting.py`  
    - Uses t-SNE to visualize the provided dataset in 2D.
  - `get_results.py`  
    - Contains the dataset class and the energy regression model (EnergyRegressor).  
  - Other support files (e.g., `sampling_algos_1.py`) for alternative approaches.

- **Task-02 Folder**  
  - `gaussianEstimate.py`  
    - Contains Gaussian Process estimation code, kernel functions, and acquisition functions.

- **PDFs and Data**  
  - Assignment specification PDFs.  
  - `A4_test_data.pt` for sample data.

## Usage

1. **Install Requirements**  
   Ensure you have PyTorch, scikit-learn, NumPy, and matplotlib installed.

2. **Run Samplers**  
   In a terminal, navigate to `TASK-0-1/` and run:  
   ```
   python sampling_algos.py
   ```
   This will print sampler timings and show a t-SNE plot.

3. **Run Plotting**  
   In the same folder, run:
   ```
   python plotting.py
   ```
   This will generate a t-SNE visualization for the dataset `A4_test_data.pt`.

4. **Run Gaussian Estimation** (Task-02):  
   Go to `Task-02/` and run:
   ```
   python gaussianEstimate.py
   ```
   (Details depend on your implementation.)

## Notes

- Ensure `trained_model_weights.pth` is in the same folder as `get_results.py` for correct energy network loading.  
- Update paths if you move code to a different directory.
