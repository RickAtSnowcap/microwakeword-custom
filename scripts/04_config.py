"""Step 4: write training_parameters.yaml exactly as the notebook does (paths made absolute).
usage: 04_config.py [v1|v2]
  v1 = notebook recipe as-is
  v2 = v1 + one extra negative feature set of confusable phrases (adversarial_augmented_features)
"""
import sys
from pathlib import Path

import yaml

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v1"

WW = Path(__file__).resolve().parent.parent  # the wakeword folder, wherever it lives
D = f"{WW}/data"
N = f"{D}/negative_datasets"
config = {}
config["window_step_ms"] = 10
TRAIN = f"{WW}/train" if VARIANT == "v1" else f"{WW}/train_{VARIANT}"
config["train_dir"] = f"{TRAIN}/trained_models/wakeword"
config["features"] = [
    {"features_dir": f"{D}/generated_augmented_features", "sampling_weight": 2.0,
     "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
    {"features_dir": f"{N}/speech", "sampling_weight": 10.0, "penalty_weight": 1.0,
     "truth": False, "truncation_strategy": "random", "type": "mmap"},
    {"features_dir": f"{N}/dinner_party", "sampling_weight": 10.0, "penalty_weight": 1.0,
     "truth": False, "truncation_strategy": "random", "type": "mmap"},
    {"features_dir": f"{N}/no_speech", "sampling_weight": 5.0, "penalty_weight": 1.0,
     "truth": False, "truncation_strategy": "random", "type": "mmap"},
    {"features_dir": f"{N}/dinner_party_eval", "sampling_weight": 0.0, "penalty_weight": 1.0,
     "truth": False, "truncation_strategy": "split", "type": "mmap"},
]
if VARIANT == "v2":
    config["features"].append(
        {"features_dir": f"{D}/adversarial_augmented_features", "sampling_weight": 2.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"})
config["training_steps"] = [10000]
config["positive_class_weight"] = [1]
config["negative_class_weight"] = [20]
config["learning_rates"] = [0.001]
config["batch_size"] = 128
config["time_mask_max_size"] = [0]
config["time_mask_count"] = [0]
config["freq_mask_max_size"] = [0]
config["freq_mask_count"] = [0]
config["eval_step_interval"] = 500
config["clip_duration_ms"] = 1500
config["target_minimization"] = 0.9
config["minimization_metric"] = None
config["maximization_metric"] = "average_viable_recall"

out = f"{TRAIN}/training_parameters.yaml"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    yaml.dump(config, f)
print("wrote", out)
