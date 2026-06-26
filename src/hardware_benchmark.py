"""
TFLite export and inference benchmark for hardware-aware NAS.
Measures latency, model size, and peak RAM during inference.
"""

import os
import time
import tempfile
import tracemalloc
from typing import Dict, Tuple

import numpy as np

import config

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def _penalty_result() -> Dict[str, float]:
    return {
        'inference_latency_ms': config.BENCHMARK_FAIL_LATENCY_MS,
        'model_size_bytes': config.BENCHMARK_FAIL_SIZE_BYTES,
        'peak_ram_mb': config.BENCHMARK_FAIL_RAM_MB,
    }


def benchmark_model(model, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Export Keras model to TFLite and benchmark inference.

    Args:
        model: Trained Keras model
        input_shape: (H, W, C) without batch dimension

    Returns:
        dict with inference_latency_ms, model_size_bytes, peak_ram_mb
    """
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp:
            tmp.write(tflite_model)
            tflite_path = tmp.name

        model_size_bytes = os.path.getsize(tflite_path)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']

        sample = np.random.randn(1, *input_shape).astype(input_dtype)

        # Warmup
        for _ in range(config.BENCHMARK_WARMUP):
            interpreter.set_tensor(input_index, sample)
            interpreter.invoke()

        tracemalloc.start()
        latencies = []
        for _ in range(config.BENCHMARK_RUNS):
            start = time.perf_counter()
            interpreter.set_tensor(input_index, sample)
            interpreter.invoke()
            interpreter.get_tensor(output_details[0]['index'])
            latencies.append((time.perf_counter() - start) * 1000.0)

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_bytes / (1024 * 1024)

        os.unlink(tflite_path)

        return {
            'inference_latency_ms': float(np.median(latencies)),
            'model_size_bytes': int(model_size_bytes),
            'peak_ram_mb': round(peak_ram_mb, 4),
        }

    except Exception as e:
        print(f"   Hardware benchmark failed: {e}")
        try:
            tracemalloc.stop()
        except Exception:
            pass
        return _penalty_result()


def apply_metrics_to_chromosome(chromosome, metrics: Dict[str, float]) -> None:
    """Store hardware benchmark metrics on a chromosome."""
    chromosome.inference_latency_ms = metrics['inference_latency_ms']
    chromosome.model_size_bytes = int(metrics['model_size_bytes'])
    chromosome.peak_ram_mb = metrics['peak_ram_mb']
