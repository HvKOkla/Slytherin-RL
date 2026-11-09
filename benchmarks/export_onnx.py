import torch
import time
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from stable_baselines3 import PPO
from games.snake.logic import SnakeLogic

# Configuration
MODEL_NAME = "Slytherin_v1"
ONNX_PATH = "benchmarks/slytherin_v1.onnx"
ONNX_QUANT_PATH = "benchmarks/slytherin_v1_int8.onnx"

def export_to_onnx():
    print("🚀 1. Exporting PyTorch model to ONNX...")
    
    # Chargement du modèle PyTorch
    model = PPO.load(MODEL_NAME)
    policy = model.policy
    
    # On crée une fausse observation (taille 10) pour tracer le réseau
    dummy_input = torch.randn(1, 10)
    
    # Export ONNX
    torch.onnx.export(
        policy.mlp_extractor, 
        dummy_input, 
        ONNX_PATH,
        input_names=["obs"],
        output_names=["features"],
        opset_version=14
    )
    
    size_mb = os.path.getsize(ONNX_PATH) / 1024
    print(f"✅ ONNX model saved: {ONNX_PATH} ({size_mb:.2f} KB)")

def quantize_model():
    print("\n🔪 2. Quantizing model to INT8...")
    
    # Quantization dynamique (poids en INT8, calculs en temps réel)
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=ONNX_QUANT_PATH,
        weight_type=QuantType.QUInt8
    )
    
    size_mb = os.path.getsize(ONNX_QUANT_PATH) / 1024
    print(f"✅ Quantized model saved: {ONNX_QUANT_PATH} ({size_mb:.2f} KB)")

def benchmark_speed():
    print("3. Benchmarking Inference Speed...")
    
    
    dummy_obs = np.random.randn(1, 10).astype(np.float32)
    
  
    model = PPO.load(MODEL_NAME)
    times_pt = []
    for _ in range(1000):
        t0 = time.perf_counter()
        model.predict(dummy_obs, deterministic=True)
        t1 = time.perf_counter()
        times_pt.append((t1 - t0) * 1000)
    
   
    session = ort.InferenceSession(ONNX_QUANT_PATH)
    input_name = session.get_inputs()[0].name
    
    times_onnx = []
    for _ in range(1000):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_obs})
        t1 = time.perf_counter()
        times_onnx.append((t1 - t0) * 1000)
    
    print(f"PyTorch (FP32)  : {np.mean(times_pt):.4f} ms")
    print(f"ONNX (INT8)     : {np.mean(times_onnx):.4f} ms")
    print(f"Speedup         : {np.mean(times_pt) / np.mean(times_onnx):.2f}x faster")

if __name__ == "__main__":
    export_to_onnx()
    quantize_model()
    benchmark_speed()
    