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
