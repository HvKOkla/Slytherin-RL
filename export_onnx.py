import torch
from stable_baselines3 import PPO

model_path = "releases/v1_1/Slytherin_v1_1"
model = PPO.load(model_path)
policy = model.policy


policy.eval()


dummy_input = torch.randn(1, 13)


torch.onnx.export(
    policy.mlp_extractor.policy_net, # On n'exporte que la partie qui prend la décision
    dummy_input,
    "slytherin_v1_1.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    dynamo=False  # <--- La virgule est importante ici !
)

print("Modèle exporté avec succès : slytherin_v1_1.onnx")