import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, List
from models.base.vision_model import VisionModel
from utils.downloader import download_joytag_model

class JoyTagModel:
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 448
        self.threshold = 0.4
        
        # Load model and tags
        self.model = self._load_model()
        self.tags = self._load_tags()
    
    @classmethod
    def from_pretrained(cls, cache_dir: str = "models") -> "JoyTagModel":
        model_path = download_joytag_model(cache_dir)
        return cls(model_path)
    
    def _load_model(self) -> VisionModel:
        model = VisionModel.load_model(self.model_path)
        model.eval()
        return model.to(self.device)
    
    def _load_tags(self) -> List[str]:
        tags_path = self.model_path / 'top_tags.txt'
        if not tags_path.exists():
            raise FileNotFoundError(f"Tags file not found at {tags_path}")
            
        with open(tags_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2
        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        if max_dim != self.image_size:
            padded_image = padded_image.resize(
                (self.image_size, self.image_size), 
                Image.Resampling.BICUBIC
            )
        
        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
        return TVF.normalize(
            image_tensor,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        image_tensor = self.prepare_image(image)
        batch = {
            'image': image_tensor.unsqueeze(0).to(self.device),
        }
        
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        scores = {
            self.tags[i]: tag_preds[0][i].item() 
            for i in range(len(self.tags))
        }
        
        predicted_tags = [
            tag for tag, score in scores.items() 
            if score > self.threshold
        ]
        tag_string = ', '.join(predicted_tags)
        
        return tag_string, scores

    def __call__(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        return self.predict(image)