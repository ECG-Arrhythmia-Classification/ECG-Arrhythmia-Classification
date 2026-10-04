# ECG Arrhythmia Classification with Deep Sequence Models

This project prepares the MIT-BIH Arrhythmia Database for ECG heartbeat classification using five classes. The data pipeline produces one common Train / Validation / Test dataset that can be used by CNN, RNN/LSTM/GRU, Transformer, and evaluation components.

## 1. Dataset

Dataset: **MIT-BIH Arrhythmia Database**

The current preprocessing maps ECG annotation symbols into five classes:

| Label | Class | Meaning |
|---:|:---:|---|
| 0 | N | Normal |
| 1 | S | Supraventricular ectopic |
| 2 | V | Ventricular ectopic |
| 3 | F | Fusion |
| 4 | Q | Unknown / paced / other |

Each heartbeat is extracted as a fixed-length segment of **180 samples**: 90 samples before the annotated heartbeat position and 90 samples after it.

## 2. Dataset statistics

The complete processed dataset contains **109,468 heartbeat samples**.

Current split:

| Split | Number of samples | Shape of X | Shape of y |
|---|---:|---|---|
| Train | 74,810 | (74810, 180) | (74810,) |
| Validation | 16,560 | (16560, 180) | (16560,) |
| Test | 18,098 | (18098, 180) | (18098,) |

The split is performed by **record ID**, so a record is assigned to only one of Train / Validation / Test.

Current class distribution:

### Train

| Class | Count | Percentage |
|:---:|---:|---:|
| N | 63,692 | 85.14% |
| S | 1,943 | 2.60% |
| V | 4,614 | 6.17% |
| F | 388 | 0.52% |
| Q | 4,173 | 5.58% |

### Validation

| Class | Count | Percentage |
|:---:|---:|---:|
| N | 13,046 | 78.78% |
| S | 338 | 2.04% |
| V | 1,006 | 6.07% |
| F | 364 | 2.20% |
| Q | 1,806 | 10.91% |

### Test

| Class | Count | Percentage |
|:---:|---:|---:|
| N | 13,870 | 76.64% |
| S | 500 | 2.76% |
| V | 1,615 | 8.92% |
| F | 50 | 0.28% |
| Q | 2,063 | 11.40% |

## 3. Project structure

```text
ECG-Arrhythmia-Classification/
│
├── data/
│   └── MIT-BIH raw ECG files
│
├── processed_data/
│   ├── X.npy
│   ├── y.npy
│   ├── record_ids.npy
│   └── split/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── X_test.npy
│       └── y_test.npy
│
├── prepare_dataset.py
├── split_dataset.py
├── verify_dataset.py
├── plot_ecg.py
├── test_ecg.py
├── data_loader.py
├── dataset.py
├── requirements.txt
└── README.md
```

## 4. Data pipeline

```text
MIT-BIH raw records
        ↓
prepare_dataset.py
        ↓
X.npy / y.npy / record_ids.npy
        ↓
split_dataset.py
        ↓
Train / Validation / Test
        ↓
verify_dataset.py
        ↓
data_loader.py
        ↓
dataset.py
        ↓
CNN / RNN-LSTM-GRU / Transformer / Evaluation
```

## 5. How to run

### Step 1: Activate the virtual environment

```bash
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the processed heartbeat dataset

```bash
python prepare_dataset.py
```

This creates:

```text
processed_data/X.npy
processed_data/y.npy
processed_data/record_ids.npy
```

### Step 4: Split the dataset

```bash
python split_dataset.py
```

This creates:

```text
processed_data/split/X_train.npy
processed_data/split/y_train.npy
processed_data/split/X_val.npy
processed_data/split/y_val.npy
processed_data/split/X_test.npy
processed_data/split/y_test.npy
```

### Step 5: Verify the processed dataset

```bash
python verify_dataset.py
```

The verification checks that the files can be loaded, the shapes are correct, all five labels are present, and there are no NaN or Inf values in the ECG data.

### Step 6: Test the common data loader

```bash
python data_loader.py
```

### Step 7: Test the Dataset wrapper

```bash
python dataset.py
```

## 6. Using the data in model code

Model members can load the common dataset through `data_loader.py`:

```python
from data_loader import load_dataset

(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
) = load_dataset()
```

The common data format is:

```text
X_train: (74810, 180)
y_train: (74810,)

X_val: (16560, 180)
y_val: (16560,)

X_test: (18098, 180)
y_test: (18098,)
```

All models should use the same Train / Validation / Test files so that CNN, RNN/LSTM/GRU, Transformer, and evaluation results are comparable.

## 7. Utility scripts

`plot_ecg.py` displays the ECG signal from a raw MIT-BIH record.

`test_ecg.py` is used to inspect a raw ECG record and its basic properties.

These scripts are for inspection/testing and are not required by model training code.

## 8. Notes for the team

- Label IDs must remain consistent: `0=N`, `1=S`, `2=V`, `3=F`, `4=Q`.
- Each heartbeat has 180 samples.
- Do not create a separate Train / Validation / Test split inside individual model files.
- Use the shared files in `processed_data/split/` through `data_loader.py` or `dataset.py`.


## 9. ECG Studio web

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

The group contract is **180 raw samples at 360 Hz** with class order **N, S, V, F, Q**. A configured trained checkpoint uses the existing `preprocessing.py` bandpass + Z-score pipeline without resampling. Browser/prototype demos use a separate illustrative 256-sample pipeline and do not report MIT-BIH accuracy.

`data_loader.py` currently reads the raw split. Models using the `team` API adapter must train with `processed_data/preprocessed/` or call the same shared `preprocess_data` function. Do not filter preprocessed signals again when sending them to the API.

- [Run the web and API](web/README.md)
- [Integrate and check trained checkpoints](web/docs/MODEL_INTEGRATION.md)
- Export one raw heartbeat: `python web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv`
- The `ECG web checks` workflow tests the frontend, backend/shared pipeline and exporter, and builds the web on working branches and pull requests. It does not publish a website or merge branches.

The generated static interface can be hosted separately when the team chooses to publish it. A trained model needs a separate HTTPS API; the web can be pointed to it through **Kết nối model**. No trained checkpoint or generated dataset is included in this web addition.

Keep web work on a separate branch such as `feature/ecg-web`; push that branch and review it with the team before merging. This addition does not deploy automatically.
