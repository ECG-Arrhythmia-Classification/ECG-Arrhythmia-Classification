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

## ECG Studio web

The web interface is in [`web/`](web/README.md). It can run independently in browser demo mode; an optional FastAPI backend connects the interface to trained models later.

From the repository root:

```bash
npm --prefix web ci
npm --prefix web run dev
```

Open `http://127.0.0.1:5173`. To run the optional backend:

```bash
python -m pip install -r web/backend/requirements.txt
python -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

The group contract is **180 raw samples at 360 Hz** with class order **N, S, V, F, Q**. The `team` API pipeline uses the existing `preprocessing.py` bandpass + Z-score pipeline without resampling; confirm that a checkpoint was trained with this preprocessing before configuring it. Browser/prototype demos use a separate illustrative 256-sample pipeline and do not report MIT-BIH accuracy.

`data_loader.py` currently reads the raw split. Models using the `team` API adapter must train with `processed_data/preprocessed/` or call the same shared `preprocess_data` function. Do not filter preprocessed signals again when sending them to the API.

- [Run the web and API](web/README.md)
- [Integrate and check trained checkpoints](web/docs/MODEL_INTEGRATION.md)
- Export one raw heartbeat: `python web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv`
- The `ECG web checks` workflow tests the frontend, backend/shared pipeline and exporter, and builds the web on working branches and pull requests. It does not publish a website or merge branches.

The repository includes the shared RAW splits through Git LFS and trained CNN/RNN weight artifacts. These training checkpoints must be loaded with their original architecture and exported to TorchScript before the web adapter can use them; the web runs explicit prototype demos until configured. See the integration guide for the checkpoint formats and pipeline requirements.

The generated static interface can be hosted separately when the team chooses to publish it. A trained model needs a separate HTTPS API; the web can be pointed to it through **Kết nối model**. The checks workflow does not deploy the website.

Keep web work on a separate branch such as `feature/ecg-web`; push that branch and review it with the team before merging. This addition does not deploy automatically.
