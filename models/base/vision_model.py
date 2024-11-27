import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import safetensors.torch
from pathlib import Path
import numpy as np

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.get('image_size', 448)
        self.num_tags = config.get('num_tags', 11824)
    
    @classmethod
    def load_model(cls, model_dir):
        model_dir = Path(model_dir)
        
        import json
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = cls(config)
        
        onnx_path = model_dir / 'model.onnx'
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        model.session = ort.InferenceSession(
            str(onnx_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        return model

    def forward(self, batch):
        image = batch['image']
        
        ort_inputs = {
            'input': image.cpu().numpy().astype(np.float32)
        }
        ort_outputs = self.session.run(None, ort_inputs)
        
        outputs = {
            'tags': torch.from_numpy(ort_outputs[0]).to(image.device)
        }
        
        return outputs