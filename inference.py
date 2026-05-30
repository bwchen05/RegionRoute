import argparse
import gc
import os

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

BASE_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

LORA_DIR = "weights/"
STYLE_TO_WEIGHTS = {
    "cyberpunk":     os.path.join(LORA_DIR, "cyberpunk.safetensors"),
    "expressionism": os.path.join(LORA_DIR, "expressionism.safetensors"),
    "line-art":      os.path.join(LORA_DIR, "line-art.safetensors"),
    "pixel-art":     os.path.join(LORA_DIR, "pixel-art.safetensors"),
}

def parse_args():
    p = argparse.ArgumentParser(description="MoE FluxKontext inference (single style per run).")
    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument("--style", required=True, choices=sorted(STYLE_TO_WEIGHTS.keys()),
                   help="Which expert (style LoRA) to activate.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    for name, path in STYLE_TO_WEIGHTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoRA weight for '{name}' not found: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Loading base pipeline: {BASE_MODEL_ID}")
    pipe = FluxKontextPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    print("Loading all four LoRA experts as named adapters...")
    for adapter_name, weight_path in STYLE_TO_WEIGHTS.items():
        pipe.load_lora_weights(
            os.path.dirname(weight_path),
            weight_name=os.path.basename(weight_path),
            adapter_name=adapter_name,
        )
        print(f"  loaded: {adapter_name}")

    print(f"Activating expert: {args.style}")
    pipe.set_adapters([args.style], adapter_weights=[1.0])

    input_image = load_image(args.image)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating: prompt={args.prompt!r}")
    result = pipe(
        prompt=args.prompt,
        image=input_image,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
