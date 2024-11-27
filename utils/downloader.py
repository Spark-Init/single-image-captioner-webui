import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import List, Dict
import shutil
import torch

def ensure_model_files(model_id: str, filenames: List[str], cache_dir: str = "models") -> Dict[str, Path]:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE", HUGGINGFACE_HUB_CACHE)
    
    repo_cache_path = Path(cache_dir).joinpath(f"models--{model_id.replace('/', '--')}")
    snapshot_path = repo_cache_path / "snapshots" / "latest"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.mkdir(exist_ok=True)
    
    downloaded_files = {}
    for filename in filenames:
        full_path = snapshot_path / Path(filename).parent
        full_path.mkdir(parents=True, exist_ok=True)
        
        output_path = snapshot_path / filename
        if not output_path.exists():
            print(f"Downloading {filename} from {model_id}...")
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=True
            )
            shutil.copy2(downloaded_path, output_path)
        
        downloaded_files[filename] = output_path
    
    missing_files = [f for f in filenames if not (snapshot_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"Failed to download or locate files: {missing_files}")
    
    return downloaded_files

def download_caption_model(cache_dir: str = "models") -> Dict[str, Path]:
    files = {
        "adapter": "wpkklhc6/image_adapter.pt",
        "config": "wpkklhc6/config.yaml"
    }
    
    downloaded_files = ensure_model_files(
        "Wi-zz/joy-caption-pre-alpha",
        list(files.values()),
        cache_dir
    )
    
    return {name: downloaded_files[path] for name, path in files.items()}

def download_joytag_model(cache_dir: str = "models") -> Path:
    required_files = [
        "config.json",
        "model.onnx",
        "model.safetensors",
        "top_tags.txt"
    ]
    
    downloaded_files = ensure_model_files(
        "fancyfeast/joytag",
        required_files,
        cache_dir
    )
    
    return next(iter(downloaded_files.values())).parent