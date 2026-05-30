#!/usr/bin/env python

import argparse
import gc
import os
import json
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers import FluxKontextPipeline

from attn_loss.loss_utils import get_word_idx

MAX_SEQ_LENGTH = 77
OUTPUT_PATH = "overfitting.parquet"

def generate_image_hash(image):
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()

def load_flux_dev_pipeline():
    id = "black-forest-labs/FLUX.1-Kontext-dev"
    text_encoder = T5EncoderModel.from_pretrained(
        id, subfolder="text_encoder_2", load_in_8bit=True, device_map="auto"
    )
    tokenizer = T5TokenizerFast.from_pretrained(id, subfolder="tokenizer_2")
    pipeline = FluxKontextPipeline.from_pretrained(
        id, text_encoder_2=text_encoder, transformer=None, vae=None, device_map="balanced"
    )
    return pipeline, tokenizer

def compute_token_positions(prompt, objects, tokenizer):
    token_positions = {}
    for obj_name in objects:
        try:
            positions = get_word_idx(prompt, obj_name, tokenizer)
            token_positions[obj_name] = positions
        except ValueError as e:
            print(f"Warning: {obj_name} not found in prompt: {prompt}")
            print(f"Error: {e}")
    return token_positions

@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_text_ids = []

    for prompt in tqdm(prompts, desc="Encoding prompts"):
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=max_sequence_length
        )

        all_prompt_embeds.append(prompt_embeds)
        all_pooled_prompt_embeds.append(pooled_prompt_embeds)
        all_text_ids.append(text_ids)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids

def load_local_dataset_with_objects(data_dir, json_file):
    import os
    from PIL import Image
    import json

    data = []

    with open(json_file, 'r', encoding='utf-8') as f:
        configs = json.load(f)
        for i, config in enumerate(configs):

            prompt = config["prompt"]
            objects = list(config['objects'].values())

            image_name = config['image_path']
            image_path = os.path.join(data_dir, image_name)
            if os.path.exists(image_path):
                image = Image.open(image_path)

                data.append({
                    "image": image,
                    "text": prompt,
                    "objects": objects,
                    "image_name": image_name
                })
    return data

def save_embeddings(data, output_path):
    df = pd.DataFrame(data)

    embedding_cols = ["prompt_embeds", "pooled_prompt_embeds", "text_ids"]
    for col in embedding_cols:
        df[col] = df[col].apply(
            lambda x: x.cpu().numpy().astype(np.float16).flatten().tolist()
            if col != "text_ids" else x.cpu().numpy().astype(np.int16).flatten().tolist()
        )

    def serialize_token_positions(token_pos_dict):
        import json
        cleaned = {}
        for key, value in token_pos_dict.items():
            if value is not None and len(value) > 0:
                if isinstance(value, np.ndarray):
                    cleaned[key] = value.tolist()
                else:
                    cleaned[key] = value
        return json.dumps(cleaned)

    df['token_positions'] = df['token_positions'].apply(serialize_token_positions)

    df.to_parquet(output_path, compression='snappy')
    print(f"Data saved to {output_path}")

def run(args):
    if args.local_dataset:
        dataset = load_local_dataset_with_objects(data_dir=args.local_dataset, json_file=args.json_file)
    else:
        dataset = load_dataset("Norod78/Yarn-art-style", split="train")
        for sample in dataset:
            sample["objects"] = []

    image_data = {}
    all_prompts = []

    for sample in dataset:
        image_hash = generate_image_hash(sample["image"])
        image_data[image_hash] = {
            "prompt": sample["text"],
            "objects": sample.get("objects", []),
            "image_name": sample.get("image_name", "")
        }
        all_prompts.append(sample["text"])

    print(f"Total samples: {len(all_prompts)}")

    pipeline, tokenizer = load_flux_dev_pipeline()

    all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )

    print("Computing token positions...")
    data = []
    for i, (image_hash, info) in enumerate(tqdm(image_data.items(), desc="Processing samples")):
        token_positions = compute_token_positions(info["prompt"], info["objects"], tokenizer)
        print(token_positions)
        data.append({
            "image_hash": image_hash,
            "prompt_embeds": all_prompt_embeds[i],
            "pooled_prompt_embeds": all_pooled_prompt_embeds[i],
            "text_ids": all_text_ids[i],
            "token_positions": token_positions,
            "prompt": info["prompt"],
            "objects": info["objects"],
        })
    print(f"Processed {len(data)} samples")

    save_embeddings(data, args.output_path)
    print(f"Data successfully serialized to {args.output_path}")

    df_test = pd.read_parquet(args.output_path)
    for i, row in df_test.iterrows():
        token_pos = row['token_positions']
        print(token_pos)

    del pipeline, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length to use for computing the embeddings.",
    )
    parser.add_argument(
        "--local_dataset",
        type=str,
        default="./dataset",
        help="Path to local dataset directory"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="./prompts.json",
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./style.parquet",
        help="Path to serialize the parquet file."
    )
    args = parser.parse_args()

    run(args)
