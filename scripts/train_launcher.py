"""Caps TensorFlow GPU memory (default 4 GB, env GPU_MEM_MB) then runs microwakeword.model_train_eval
with the given CLI args, so other GPU work (a speech service, say) keeps the rest of the card."""
import os, sys, runpy
os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get("GPU_MEM_MB", "4096")))])
print("GPUs:", gpus, flush=True)
sys.argv = ["microwakeword.model_train_eval"] + sys.argv[1:]
runpy.run_module("microwakeword.model_train_eval", run_name="__main__")
