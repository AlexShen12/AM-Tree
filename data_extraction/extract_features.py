from PIL import Image
import requests
import torch
import transformers
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import os
from pprint import pprint
import pdb

import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode



def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)



# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def clip_es():
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True).to('cuda').eval()

    base_path = Path('/users/a/l/alshen/VideoTree/VideoTree/data/VideoMME_frames')
    save_folder = '/users/a/l/alshen/VideoTree/VideoTree/data/VideoMME_feature'

    example_path_list = sorted(p for p in base_path.iterdir() if p.is_dir())

    already_done = {Path(f).stem for f in os.listdir(save_folder) if f.endswith('.pt')}
    pending = [p for p in example_path_list if p.name not in already_done]
    print(f"Resuming: {len(already_done)} already done, {len(pending)} remaining.")

    pbar = tqdm(total=len(pending))

    for example_path in pending:
        image_paths = list(example_path.iterdir())
        image_paths.sort(key=lambda x: int(x.stem))
        img_feature_list = []
        for image_path in image_paths:
            try:
                image = Image.open(str(image_path))
                image.verify()
                image = Image.open(str(image_path))
            except Exception as e:
                print(f"Skipping corrupt frame {image_path}: {e}")
                continue

            input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

            with torch.no_grad(), torch.amp.autocast('cuda'):
                image_features = model.encode_image(input_pixels)
                img_feature_list.append(image_features)

        if not img_feature_list:
            print(f"No valid frames for {example_path.name}, skipping.")
            pbar.update(1)
            continue

        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1)

        save_image_features(img_feats, example_path.name, save_folder)
        pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    clip_es()
