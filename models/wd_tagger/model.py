# models/wd_tagger/model.py
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
from .config import *

class WDTaggerModel:
    
    def __init__(self, model_variant: str):
        if model_variant not in MODELS:
            raise ValueError(f"Unknown model variant: {model_variant}. Available variants: {list(MODELS.keys())}")
            
        self.model_variant = model_variant
        self.model_config = MODELS[model_variant]
        self.model = None
        self.tags_df = None
        self.target_size = self.model_config['target_size']
        self.threshold = 0.4
        
    @classmethod
    def from_pretrained(cls, model_variant: str = 'wd-swinv2-v3', cache_dir: str = "models") -> "WDTaggerModel":
        instance = cls(model_variant)
        instance._load_model(cache_dir)
        return instance
    
    def _load_model(self, cache_dir: str):
        repo_id = self.model_config['repo_id']
        
        model_path = huggingface_hub.hf_hub_download(repo_id, "model.onnx")
        tags_path = huggingface_hub.hf_hub_download(repo_id, "selected_tags.csv")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        self.model = rt.InferenceSession(
            model_path,
            providers=providers,
            provider_options=[{'device_id': 0}] if torch.cuda.is_available() else None
        )
        
        self.tags_df = pd.read_csv(tags_path)
        
        self.rating_tags = self.tags_df[self.tags_df['category'].isin(RATING_CATEGORIES)].index.tolist()
        self.general_tags = self.tags_df[self.tags_df['category'].isin(GENERAL_CATEGORIES)].index.tolist()
        self.character_tags = self.tags_df[self.tags_df['category'].isin(CHARACTER_CATEGORIES)].index.tolist()
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def predict(self, image: Image.Image, general_threshold: float = DEFAULT_GENERAL_THRESHOLD, 
                character_threshold: float = DEFAULT_CHARACTER_THRESHOLD) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        image_tensor = self._prepare_image(image)
        
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        predictions = self.model.run([output_name], {input_name: image_tensor})[0][0]
        
        results = {
            'rating': self._process_ratings(predictions),
            'general': self._process_general_tags(predictions, general_threshold),
            'characters': self._process_character_tags(predictions, character_threshold)
        }
                
        return results

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_w, pad_h))
        
        if max_dim != self.target_size:
            padded = padded.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
            
        img_array = np.array(padded, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        
        return np.expand_dims(img_array, axis=0)

    def _process_ratings(self, predictions: np.ndarray) -> List[Tuple[str, float]]:
        rating_scores = [(self.tags_df.iloc[i]['name'], float(predictions[i])) 
                        for i in self.rating_tags]
        return sorted(rating_scores, key=lambda x: x[1], reverse=True)

    def _process_general_tags(self, predictions: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        general_scores = [(self.tags_df.iloc[i]['name'], float(predictions[i])) 
                         for i in self.general_tags 
                         if predictions[i] > threshold]
        return sorted(general_scores, key=lambda x: x[1], reverse=True)

    def _process_character_tags(self, predictions: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        character_scores = [(self.tags_df.iloc[i]['name'], float(predictions[i])) 
                          for i in self.character_tags 
                          if predictions[i] > threshold]
        return sorted(character_scores, key=lambda x: x[1], reverse=True)