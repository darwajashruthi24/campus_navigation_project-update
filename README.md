 # Campus Navigation — Frontend + Backend

## Overview
This repository contains a computer-vision project for campus navigation. It includes training notebooks, model weights, and a simple Gradio frontend to run detection and classification locally.

- Frontend: `gradio_app.py` — a Gradio app that runs YOLO detection and ResNet classification on uploaded images.
- Backend / Training: Jupyter notebooks in the repository used to prepare data and train models (YOLO and ResNet): `01_setup_and_folders.ipynb`, `02_train_yolo.ipynb`, `03_train_resnet50.ipynb`, `04_evaluate_resnet50.ipynb`, `05_navigate_demo.ipynb`, `06_improvements.ipynb`.
- Model weights (examples provided): `yolov8s.pt`, `resnet50_campus.pth`, `resnet50_campus_improved.pth`, plus Ultralytics run outputs under `runs/`.

## Files of interest
- `gradio_app.py` — Launches Gradio UI for image upload, runs YOLO and/or ResNet inference, and returns annotated image + JSON results.
- `requirements-gradio.txt` — Python packages needed to run the frontend locally.
- `data/` — Image folders organized by class (train/val/test) used by the ResNet training pipeline.
- `data_yolo/` — YOLO-formatted dataset (images + labels + `data.yaml`).
- `obj_train_data/` and `runs/` — CVAT export and YOLO training outputs respectively.

## Quick start — Frontend (local)
1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r "requirements-gradio.txt"
```

3. Place model files (if you have them) in the project root:
- YOLO weights: `yolov8s.pt` or your custom weights from `runs/.../weights/best.pt`
- ResNet weights: `resnet50_campus_improved.pth` or `resnet50_campus.pth`

4. Run the frontend app:

```powershell
python "gradio_app.py"
```

Open the local Gradio URL printed in the terminal (usually `http://127.0.0.1:7860`).

## Backend — Training & Evaluation
- Use the notebooks to reproduce dataset preparation and training steps:
  - `01_setup_and_folders.ipynb` — environment and folder setup
  - `02_train_yolo.ipynb` — trains YOLOv8 detector (Ultralytics)
  - `03_train_resnet50.ipynb` — trains a ResNet50 classifier on cropped images or full images
  - `04_evaluate_resnet50.ipynb` — evaluation and metrics

Notes:
- Notebooks expect data in `obj_train_data/`, `data_yolo/`, and `data/` as shown in the project structure.
- Training on CPU can be slow — prefer a CUDA-enabled machine when available.

## Troubleshooting
- If `gradio_app.py` fails with missing packages, re-run `pip install -r requirements-gradio.txt` inside the activated venv.
- If models are not found, verify the file names and put them in the repository root or update the `gradio_app.py` paths.
- For large models on CPU, inference will be slow; select `cpu` device in the app or run on a GPU-enabled environment.

## Next steps I can do for you
- Add a `Dockerfile` to containerize the frontend.
- Commit files and create a small CI workflow.
- Improve the UI (batch upload, video stream, or camera input).
