# Dataset

## Source

We use the [MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/) from PhysioNet. It contains 48 two-channel ECG recordings with reference annotations, sampled at 360 Hz. The raw recordings are not included in this repository.

## Download the processed data

The processed train, validation, and test files are stored with [Git LFS](https://git-lfs.com/). Install Git LFS, then run:

```bash
git lfs install
git clone https://github.com/ECG-Arrhythmia-Classification/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
git lfs pull
```

The arrays will be in `processed_data/split/`. Check them with:

```bash
python verify_dataset.py
```

## How the data was prepared

[`prepare_dataset.py`](./prepare_dataset.py) reads all 48 records listed in the script and uses the first ECG channel. For each included annotation, it extracts a 180-sample heartbeat segment with 90 samples before the annotation and 90 samples after the annotation. Windows that run past the start or end of a record are skipped. The script does not filter, resample, or normalize the signal.

Annotations are grouped into five classes:

| Label | Class | Annotation symbols |
|---:|:---:|---|
| 0 | N | `N`, `L`, `R`, `e`, `j` |
| 1 | S | `A`, `a`, `J`, `S` |
| 2 | V | `V`, `E` |
| 3 | F | `F` |
| 4 | Q | `Q`, `/`, `f` |

[`split_dataset.py`](./split_dataset.py) assigns whole records to one split, so beats from a record cannot appear in more than one split.

| Split | Records | Heartbeats | `X` shape | `y` shape |
|---|---:|---:|---|---|
| Train | 35 | 74,810 | `(74810, 180)` | `(74810,)` |
| Validation | 6 | 16,560 | `(16560, 180)` | `(16560,)` |
| Test | 7 | 18,098 | `(18098, 180)` | `(18098,)` |

Record IDs:

- **Train:** `100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 119, 121, 122, 123, 124, 200, 201, 202, 207, 208, 212, 214, 219, 220, 221, 228, 230, 231, 232, 234`
- **Validation:** `118, 203, 213, 215, 217, 222`
- **Test:** `104, 114, 205, 209, 210, 223, 233`

The processed dataset contains 109,468 heartbeats in total.

## Rebuild the data

To create the processed files yourself, download version 1.0.0 from the [PhysioNet dataset page](https://physionet.org/content/mitdb/1.0.0/) and put the `.dat`, `.hea`, and `.atr` record files in the repository's `data/` folder. Install the dependencies and run the scripts from the repository root:

```bash
pip install -r requirements.txt
python prepare_dataset.py
python split_dataset.py
python verify_dataset.py
```

This creates `processed_data/X.npy`, `processed_data/y.npy`, `processed_data/record_ids.npy`, and the six split arrays in `processed_data/split/`.

## Load the data

Use the shared loader in model code:

```python
from data_loader import load_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
```

Use these same splits when comparing models.

## Citation

Please cite the dataset and its original papers:

- Moody GB, Mark RG. “The MIT-BIH Arrhythmia Database on CD-ROM and software for use with it.” *Computers in Cardiology*. 1990;17:185–188.
- Mark RG, Schluter PS, Moody GB, Devlin PH, Chernoff D. “An annotated ECG database for evaluating arrhythmia detectors.” *IEEE Transactions on Biomedical Engineering*. 1982;29(8):600.
- [MIT-BIH Arrhythmia Database on PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
