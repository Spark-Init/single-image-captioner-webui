from flask import Flask, render_template, request, jsonify
import base64, io
import logging
import warnings
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.joy_caption.model import JoyCaptionModel
from models.joy_tag.model import JoyTagModel
from models.wd_tagger.model import WDTaggerModel
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

app = Flask(__name__)

loaded_models = {
    'caption_model': None,
    'joytag_model': None,
    'wd_tagger_model': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        model_type = request.json['model_type']
        
        if model_type == 'caption':
            if loaded_models['caption_model'] is None:
                loaded_models['caption_model'] = JoyCaptionModel.from_pretrained()
                return jsonify({'status': 'success', 'message': 'Caption model loaded'})
            return jsonify({'status': 'info', 'message': 'Caption model already loaded'})
            
        elif model_type == 'joytag':
            if loaded_models['joytag_model'] is None:
                loaded_models['joytag_model'] = JoyTagModel.from_pretrained()
                return jsonify({'status': 'success', 'message': 'JoyTag model loaded'})
            return jsonify({'status': 'info', 'message': 'JoyTag model already loaded'})
            
        elif model_type == 'wdtagger':
            model_variant = request.json.get('model_variant', 'wd-swinv2-v3')
            if loaded_models['wd_tagger_model'] is None:
                loaded_models['wd_tagger_model'] = WDTaggerModel.from_pretrained(model_variant)
                return jsonify({'status': 'success', 'message': f'WD Tagger ({model_variant}) loaded'})
            return jsonify({'status': 'info', 'message': 'WD Tagger model already loaded'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/unload_model', methods=['POST'])
def unload_model():
    try:
        model_type = request.json['model_type']
        
        if model_type == 'caption' and loaded_models['caption_model'] is not None:
            del loaded_models['caption_model']
            loaded_models['caption_model'] = None
            torch.cuda.empty_cache()
            return jsonify({'status': 'success', 'message': 'Caption model unloaded'})
            
        elif model_type == 'joytag' and loaded_models['joytag_model'] is not None:
            del loaded_models['joytag_model']
            loaded_models['joytag_model'] = None
            torch.cuda.empty_cache()
            return jsonify({'status': 'success', 'message': 'JoyTag model unloaded'})
            
        elif model_type == 'wdtagger' and loaded_models['wd_tagger_model'] is not None:
            loaded_models['wd_tagger_model'].unload()
            loaded_models['wd_tagger_model'] = None
            return jsonify({'status': 'success', 'message': 'WD Tagger model unloaded'})
            
        return jsonify({'status': 'info', 'message': f'Model {model_type} not loaded'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json['image'].split(',')[1]
        model_type = request.json.get('model_type', 'caption')
        
        if model_type == 'caption' and loaded_models['caption_model'] is None:
            return jsonify({'error': 'Caption model not loaded'}), 400
        elif model_type == 'joytag' and loaded_models['joytag_model'] is None:
            return jsonify({'error': 'JoyTag model not loaded'}), 400
        elif model_type == 'wdtagger' and loaded_models['wd_tagger_model'] is None:
            return jsonify({'error': 'WD Tagger model not loaded'}), 400
        
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if model_type == 'joytag':
            tag_string, scores = loaded_models['joytag_model'].predict(img)
            return jsonify({
                'caption': tag_string,
                'scores': scores
            })
        elif model_type == 'wdtagger':
            general_threshold = float(request.json.get('general_threshold', 0.35))
            character_threshold = float(request.json.get('character_threshold', 0.85))
            
            results = loaded_models['wd_tagger_model'].predict(
                img,
                general_threshold=general_threshold,
                character_threshold=character_threshold
            )
            
            return jsonify({
                'results': results,
                'caption': ', '.join(tag for tag, _ in results['general'])
            })
        else:
            caption = loaded_models['caption_model'].generate_caption(img)
            return jsonify({'caption': caption})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    return jsonify({
        'caption_model': loaded_models['caption_model'] is not None,
        'joytag_model': loaded_models['joytag_model'] is not None,
        'wd_tagger_model': loaded_models['wd_tagger_model'] is not None
    })

def main():
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()