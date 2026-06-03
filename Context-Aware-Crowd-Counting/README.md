# Context-Aware Crowd Counting with Domain Adaptation for Wheat Head Counting

This repository is a PyTorch implementation for **Context-Aware Crowd Counting (CACC)** adapted to the **Global Wheat Head Detection (GWHD 2021)** dataset. It extends the baseline CACC architecture (originally introduced by Weizhe Liu et al. in CVPR 2019) with **Gradient Reversal Layer (GRL)** and **DeepCORAL** domain adaptation mechanisms to generalize wheat head counting across diverse country domains.

To guarantee complete reproducibility and eliminate library version conflicts, the entire training, testing, and evaluation pipeline is fully dockerized.

---

## Table of Contents
1. [Dataset Acquisition & Preparation](#1-dataset-acquisition--preparation)
2. [Recommended Setup: Docker](#2-recommended-setup-docker)
3. [Training Experiments inside Docker](#3-training-experiments-inside-docker)
4. [Testing & Inference inside Docker](#4-testing--inference-inside-docker)
5. [Reproducing Thesis Results & Visualizations](#5-reproducing-thesis-results--visualizations)
6. [Secondary Option: Manual Local Setup (Without Docker)](#secondary-option-manual-local-setup-without-docker)
7. [Repository File Guide](#repository-file-guide)
8. [Citations](#citations)

---

## 1. Dataset Acquisition & Preparation

### A. Download GWHD 2021
The dataset can be downloaded from Zenodo:
* **Zenodo Repository**: [Global Wheat Head Detection Dataset 2021](https://zenodo.org/records/5092309)

Download the images zip and the annotations/metadata CSV files.

### B. Directory Structure
Create a dataset folder (e.g. `gwhd_2021`) and extract the dataset so it follows this layout:
```text
gwhd_2021/
├── metadata.csv        # Maps sub-domains to countries
├── train.csv           # Training bounding boxes
├── val.csv             # Validation bounding boxes
├── test.csv            # Test bounding boxes
└── images/             # Folder containing all raw PNG/JPG images
    ├── image1.png
    ├── image2.png
    ...
```

---

## 2. Recommended Setup: Docker

Building the codebase via Docker ensures that all dependencies (including PyTorch, torchvision, and OpenCV system packages) compile and run exactly as they did in the thesis experiments.

### Step A: Build the Docker Image
In your terminal, navigate to the project directory and run:
```bash
docker build -t crowd-counting .
```

### Step B: Start the Container
Start the container while mounting your local dataset directory (`gwhd_2021`) to the container's workspace to keep the image lightweight and avoid copying dataset files:
```bash
docker run --gpus all -it \
  -v /absolute/path/to/gwhd_2021:/workspace/gwhd_2021 \
  crowd-counting
```
*(If you are running on a CPU-only host or Apple Silicon without NVIDIA GPUs, you can omit the `--gpus all` flag).*

---

## 3. Training Experiments inside Docker

Once inside the Docker container shell, you can preprocess datasets and train models.

### Step A: Preprocess Dataset & Create Density Maps
Convert bounding boxes to HDF5 density maps and generate the image list JSON files:
```bash
# 1. Generate train density maps
python make_dataset.py --csv_file gwhd_2021/train.csv --img_folder gwhd_2021/images --output_folder gwhd_2021/ground_truth/train

# 2. Generate val density maps
python make_dataset.py --csv_file gwhd_2021/val.csv --img_folder gwhd_2021/images --output_folder gwhd_2021/ground_truth/val

# 3. Create JSON listings
python create_json.py --h5_folder gwhd_2021/ground_truth/train --img_folder gwhd_2021/images --output_json train.json
python create_json.py --h5_folder gwhd_2021/ground_truth/val --img_folder gwhd_2021/images --output_json val.json
```

### Step B: Train Model
Choose the baseline or one of the domain adaptation configurations:

#### Baseline CACC (No Domain Adaptation)
```bash
python train.py train.json val.json --algorithm none --epochs 100 --exp_name baseline
```

#### Gradient Reversal Layer (GRL)
Attach a domain classification head with GRL. Location options: `frontend`, `context`, or `concat`:
```bash
python train.py train.json val.json \
  --algorithm grl \
  --grl_location concat \
  --lambda_domain 0.0001 \
  --epochs 100 \
  --exp_name grl_concat
```

#### DeepCORAL Domain Generalization
Align feature distributions. Location options: `frontend`, `context`, or `concat`:
```bash
python train.py train.json val.json \
  --algorithm coral \
  --coral_location frontend \
  --lambda_domain 0.0001 \
  --epochs 100 \
  --exp_name coral_frontend
```

---

## 4. Testing & Inference inside Docker

To evaluate a single saved model checkpoint on the test set:
```bash
python test.py path/to/test.json \
  --model_path path/to/model_best.pth.tar \
  --grl_location none \
  --batch_size 1
```

---

## 5. Reproducing Thesis Results & Visualizations

We provide automated scripts to evaluate models and recreate the charts and figures from the thesis paper:

1. **Batch Evaluation**: Evaluates the baseline, GRL, and CORAL models on the test set and outputs `test_results.json`:
   ```bash
   python run_batch_tests.py
   ```
2. **Compile Results & Plots**: Parses validation histories and prints summary tables and convergence curves:
   ```bash
   python create_thesis_results.py
   ```
   *Generates `thesis_results_table.md`, `thesis_long_runs_comparison.png`, `thesis_grl_grid_search.png`, and `thesis_adaptation_convergence_5_epochs.png`*.
3. **Compare Predictions & Annotations**: Generates side-by-side comparative visualizations comparing prediction maps against ground truth annotations:
   ```bash
   python generate_thesis_visualizations.py
   ```

---

## Secondary Option: Manual Local Setup (Without Docker)

If you prefer to run the code using the normal Python environment directly on your host machine without Docker:

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install exact dependency versions**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Repository File Guide

* **[model.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/model.py)**: Network definitions including CANNet, the Scale-Aware Contextual module, GRL head, and DeepCORAL features.
* **[train.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/train.py)**: Training script supporting GRL, DeepCORAL, and custom logging/curve plotting.
* **[test.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/test.py)**: Evaluation script for test datasets.
* **[dataset.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/dataset.py)**: PyTorch list dataset class with sub-domain metadata grouping.
* **[image.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/image.py)**: Data augmentation and pre-processing functions.
* **[make_dataset.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/make_dataset.py)**: Generates ground-truth density maps from coordinates.
* **[create_json.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/create_json.py)**: Generates list JSON paths for the training data pipeline.
* **[utils.py](file:///Volumes/JULIANSSD/Code/UniCode/Year3/Thesis/JulianThesis/Context-Aware-Crowd-Counting/utils.py)**: Helper functions for checkpoint saving.

---

## Citations

If you use this code or the CACC framework, please cite the original papers:

```bibtex
@InProceedings{Liu_2019_CVPR,
  author = {Liu, Weizhe and Salzmann, Mathieu and Fua, Pascal},
  title = {Context-Aware Crowd Counting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

```bibtex
@article{david_gwhd_2021,
  title={Global Wheat Head Detection 2021: An Improved Dataset for Benchmarking Wheat Head Detection Methods},
  author={David, Etienne and others},
  journal={Plant Phenomics},
  year={2021}
}
```
