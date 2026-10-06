from pathlib import Path
import json
import csv


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    BASE_DIR
    / "results"
    / "evaluation"
    / "evaluation_results.json"
)

BENCHMARK_FILE = (
    BASE_DIR
    / "results"
    / "benchmark"
    / "benchmark_results.json"
)

ROBUSTNESS_FILE = (
    BASE_DIR
    / "results"
    / "robustness"
    / "robustness_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "model_comparison.csv"
)


# ==================================================
# LOAD JSON
# ==================================================

def load_json(path):

    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ==================================================
# SAFE GET
# ==================================================

def safe_get(
    data,
    *keys,
    default=""
):

    current = data

    for key in keys:

        if not isinstance(
            current,
            dict
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ==================================================
# BUILD MODEL ROW
# ==================================================

def build_model_row(
    model_name,
    evaluation_data,
    benchmark_data,
    robustness_data
):

    row = {
        "Model": model_name,

        # Evaluation
        "Accuracy":
            safe_get(
                evaluation_data,
                model_name,
                "accuracy"
            ),

        "Macro_Precision":
            safe_get(
                evaluation_data,
                model_name,
                "precision_macro"
            ),

        "Macro_Recall":
            safe_get(
                evaluation_data,
                model_name,
                "recall_macro"
            ),

        "Macro_F1":
            safe_get(
                evaluation_data,
                model_name,
                "f1_macro"
            ),

        "Weighted_F1":
            safe_get(
                evaluation_data,
                model_name,
                "f1_weighted"
            ),

        # Benchmark
        "Parameters":
            safe_get(
                benchmark_data,
                model_name,
                "trainable_parameters"
            ),

        "Checkpoint_MB":
            safe_get(
                benchmark_data,
                model_name,
                "checkpoint_size_mb"
            ),

        "Total_Inference_Time_sec":
            safe_get(
                benchmark_data,
                model_name,
                "total_inference_time_sec"
            ),

        "Inference_ms_per_sample":
            safe_get(
                benchmark_data,
                model_name,
                "avg_inference_time_per_sample_ms"
            ),

        "Throughput_samples_per_sec":
            safe_get(
                benchmark_data,
                model_name,
                "throughput_samples_per_sec"
            ),

        # Robustness - clean
        "Clean_Accuracy":
            safe_get(
                robustness_data,
                model_name,
                "clean",
                "accuracy"
            ),

        "Clean_Macro_F1":
            safe_get(
                robustness_data,
                model_name,
                "clean",
                "macro_f1"
            ),

        # 30 dB
        "Accuracy_30dB":
            safe_get(
                robustness_data,
                model_name,
                "low_30db",
                "accuracy"
            ),

        "Macro_F1_30dB":
            safe_get(
                robustness_data,
                model_name,
                "low_30db",
                "macro_f1"
            ),

        # 20 dB
        "Accuracy_20dB":
            safe_get(
                robustness_data,
                model_name,
                "medium_20db",
                "accuracy"
            ),

        "Macro_F1_20dB":
            safe_get(
                robustness_data,
                model_name,
                "medium_20db",
                "macro_f1"
            ),

        # 10 dB
        "Accuracy_10dB":
            safe_get(
                robustness_data,
                model_name,
                "high_10db",
                "accuracy"
            ),

        "Macro_F1_10dB":
            safe_get(
                robustness_data,
                model_name,
                "high_10db",
                "macro_f1"
            ),
    }

    return row


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("CREATE MODEL COMPARISON SUMMARY")
    print("=" * 70)

    evaluation_data = load_json(
        EVALUATION_FILE
    )

    benchmark_data = load_json(
        BENCHMARK_FILE
    )

    robustness_data = load_json(
        ROBUSTNESS_FILE
    )

    model_names = [
        "CNN",
        "RNN_LSTM",
        "Transformer"
    ]

    rows = []

    for model_name in model_names:

        row = build_model_row(
            model_name,
            evaluation_data,
            benchmark_data,
            robustness_data
        )

        rows.append(row)

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print("\nSummary created successfully:")
    print(OUTPUT_CSV)

    print("\nModels:")
    for row in rows:
        print(
            f"- {row['Model']}"
        )


if __name__ == "__main__":
    main()