import torch
import torch.amp.autocast_mode
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from typing import List, Union, Dict

from .config import *
from .adapter import ImageAdapter
from ..base.vision_model import VisionModel 
from utils.downloader import download_caption_model

class JoyCaptionModel:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()

    @classmethod
    def from_pretrained(cls, cache_dir: str = "models"):
        model = cls()
        model_paths = download_caption_model(cache_dir)
        model.load_adapter(model_paths['adapter'])
        return model

    def _load_models(self):
        print("Loading CLIP 📎")
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model.eval().requires_grad_(False).to(self.device)
        
        print("Loading tokenizer 🪙")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        
        print("Loading LLM 🤖")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        ).eval()
        
        self.image_adapter = None

    def load_adapter(self, adapter_path: str):
        print("Loading image adapter 🖼️")
        self.image_adapter = ImageAdapter(
            self.clip_model.config.hidden_size,
            self.text_model.config.hidden_size
        )
        self.image_adapter.load_state_dict(
            torch.load(adapter_path, map_location="cpu", weights_only=True)
        )
        self.image_adapter.eval().to(self.device)

    @torch.no_grad()
    def generate_caption(self, image: Union[Image.Image, List[Image.Image]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(image, Image.Image):
            images = [image]
            single = True
        else:
            images = image
            single = False
            
        torch.cuda.empty_cache()
        all_captions = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            try:
                image_inputs = self.clip_processor(images=batch, return_tensors='pt', padding=True)
                pixel_values = image_inputs.pixel_values.to(self.device)

                with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                    vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                    image_features = vision_outputs.hidden_states[-2]
                    embedded_images = self.image_adapter(image_features).to(dtype=torch.bfloat16)

                prompt = self.tokenizer.encode(VLM_PROMPT, return_tensors='pt')
                prompt_embeds = self.text_model.model.embed_tokens(prompt.to(self.device)).to(dtype=torch.bfloat16)
                embedded_bos = self.text_model.model.embed_tokens(
                    torch.tensor([[self.tokenizer.bos_token_id]], device=self.device, dtype=torch.int64)
                ).to(dtype=torch.bfloat16)

                inputs_embeds = torch.cat([
                    embedded_bos.expand(embedded_images.shape[0], -1, -1),
                    embedded_images,
                    prompt_embeds.expand(embedded_images.shape[0], -1, -1),
                ], dim=1).to(dtype=torch.bfloat16)

                input_ids = torch.cat([
                    torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
                    torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
                    prompt.expand(embedded_images.shape[0], -1),
                ], dim=1).to(self.device)

                attention_mask = torch.ones_like(input_ids)

                generate_ids = self.text_model.generate(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=300,
                    do_sample=True,
                    top_k=10,
                    temperature=0.5,
                )

                generate_ids = generate_ids[:, input_ids.shape[1]:]
                for ids in generate_ids:
                    caption = self.tokenizer.decode(
                        ids[:-1] if ids[-1] == self.tokenizer.eos_token_id else ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    caption = caption.replace('<|end_of_text|>', '').replace('<|finetune_right_pad_id|>', '').strip()
                    all_captions.append(caption)

            except Exception as e:
                print(f"Error processing batch: {e}")
                print("Skipping this batch and continuing...")
                continue

        return all_captions[0] if single else all_captions

    def __call__(self, image: Union[Image.Image, List[Image.Image]], batch_size: int = 1) -> Union[str, List[str]]:
        return self.generate_caption(image, batch_size)