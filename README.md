# ECG Arrhythmia Classification

This project uses the MIT-BIH Arrhythmia Database to classify ECG heartbeats into five classes. The shared dataset is split into train, validation, and test sets by record, so heartbeats from the same record do not appear in different splits.

## Dataset

Each heartbeat is represented by 180 samples: 90 before and 90 after its annotated position.

| Label | Class | Description |
|---:|:---:|---|
| 0 | N | Normal |
| 1 | S | Supraventricular ectopic |
| 2 | V | Ventricular ectopic |
| 3 | F | Fusion |
| 4 | Q | Paced, unknown, or other |

The processed dataset has 109,468 heartbeats:

| Split | Samples | Shape of X | Shape of y |
|---|---:|---|---|
| Train | 74,810 | `(74810, 180)` | `(74810,)` |
| Validation | 16,560 | `(16560, 180)` | `(16560,)` |
| Test | 18,098 | `(18098, 180)` | `(18098,)` |

## Get the ready-to-use dataset

The files in `processed_data/split/` are stored with [Git LFS](https://git-lfs.com/).

1. Install Git and Git LFS. Follow the [Git LFS installation guide](https://git-lfs.com/) for your operating system.
2. Clone the repository and download the data:

   ```bash
   git lfs install
   git clone https://github.com/ECG-Arrhythmia-Classification/ECG-Arrhythmia-Classification.git
   cd ECG-Arrhythmia-Classification
   git lfs pull
   ```

3. Create a Python environment and install the dependencies:

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Check that the dataset files are present and valid:

   ```bash
   python verify_dataset.py
   ```

If Git LFS was not installed before cloning, install it and run `git lfs pull` from the repository folder.

## Use the data

The ready-to-use files are:

```text
processed_data/split/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
└── y_test.npy
```

Load all three splits in Python with:

```python
from data_loader import load_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
```

Use these shared splits for all models so their results can be compared fairly. Do not create a different train/validation/test split for each model.

## Train the Transformer model

Install the dependencies from `requirements.txt`, then create normalized data and class weights before training:

```bash
python preprocessing.py
python augmentation.py
python train_transformer.py
```

The trainer reads normalized splits from `processed_data/preprocessed/` and class weights from `processed_data/augmented/class_weights.npy`. It uses the class weights with the original train split to account for rare classes. Edit the settings at the top of `train_transformer.py` to change the model or training parameters.

The best checkpoint is selected by validation macro-F1. Results are written to `results/transformer/`: `best_model.pt`, `metrics.json`, `training_history.csv`, `training_curve.png`, and `attention_example.png`. The curve and CSV update after each epoch.

## Rebuild the processed dataset

Most team members do not need to run these steps. The ready-to-use split files are already available through Git LFS.

To rebuild the dataset, first put the MIT-BIH raw record files (`.dat`, `.hea`, and `.atr`) in the `data/` folder. Raw files are not included in this repository. Then run:

```bash
python prepare_dataset.py
python split_dataset.py
python verify_dataset.py
```

This creates `processed_data/X.npy`, `processed_data/y.npy`, `processed_data/record_ids.npy`, and the split files in `processed_data/split/`.

## Notes

- Keep the label mapping unchanged: `0=N`, `1=S`, `2=V`, `3=F`, `4=Q`.
- `plot_ecg.py` and `test_ecg.py` are utilities for inspecting raw ECG records.
- `requirements.txt` lists the Python dependencies.
