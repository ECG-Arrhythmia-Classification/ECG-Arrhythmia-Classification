# English interface terminology

The interface uses English (US number formatting) and sentence case for labels,
headings, actions, and messages. Model names and class codes retain their original
capitalization. All visible copy, accessible names, tooltips, JSON previews,
validation messages, and the downloadable architecture diagram are in English.

| Term | Meaning in this project |
|---|---|
| Heartbeat | A single ECG segment used for classification; not a full recording |
| Raw signal | Input before filtering and normalization |
| Preprocessed signal | Signal after the pipeline actually used for inference |
| Ground truth | Dataset reference label, used only for comparison |
| Softmax score | Model output for a class; not a calibrated clinical probability |
| Illustrative score | Score from the untrained demonstration prototype |
| Checkpoint | Saved model weights |
| Inference time | Time for the current model inference call |
| Macro F1 | F1 averaged with equal weight for each class |
| Weighted F1 | F1 averaged using each class's support |
| Confusion matrix | Rows are actual labels; columns are predicted labels |
| Noise robustness | Saved performance under Gaussian noise at specified SNR levels |

## Heartbeat classes

Names follow the grouping implemented in `prepare_dataset.py`, rather than treating
each group as a single original annotation symbol:

- **N — Normal beat group:** N, L, R, e, j; includes bundle branch block and escape beats.
- **S — Supraventricular ectopic beat:** A, a, J, S.
- **V — Ventricular ectopic beat:** V, E; includes ventricular escape beats.
- **F — Fusion beat:** F, fusion of ventricular and normal beats.
- **Q — Unclassifiable / paced beat:** Q, /, f; includes paced–normal fusion beats.

Original annotation meanings: [PhysioNet standard annotations](https://physionet.org/static/lightwave/doc/annotations.html).
Class N does not establish that a patient is healthy, and signal noise alone does
not define class Q.

## API compatibility

The backend retains legacy `label_vi` fields for existing API clients and supplies
`label_en` on class payloads. The frontend uses the project class code to normalize
English labels, so it also works with older API payloads that only contain
`label_vi`. English prediction exports omit legacy Vietnamese label fields;
class codes, numeric scores, waveforms, and checkpoint metadata are preserved.

This language change does not retrain models or alter preprocessing or predictions.
