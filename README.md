# RegionRoute
The official repository of the CVPR 2026 paper "RegionRoute: Regional Style Transfer with Diffusion Model".

## Environment

Create the venv
```bash
python3.9 -m venv flux_env && source flux_env/bin/activate &&
pip install -r requirements.txt
```

## Pretrained weights

Download the four style LoRAs from the GitHub release into `weights/`:

```bash
gh release download v1.0 -R bwchen05/RegionRoute -D weights/ -p '*.safetensors'
```

Or without the `gh` CLI:

```bash
mkdir -p weights
for f in cyberpunk expressionism line-art pixel-art; do
    wget -P weights/ "https://github.com/bwchen05/RegionRoute/releases/download/v1.0/${f}.safetensors"
done
```

After download, `weights/` should contain `cyberpunk.safetensors`,
`expressionism.safetensors`, `line-art.safetensors`, `pixel-art.safetensors`
(~18 MB each).

## Inference

```bash
python inference.py \
    --image  ./context_img.png \
    --prompt "make the man in pixel-art style, keep other area unchanged" \
    --style  pixel-art \
    --output ./out.png
```

`--style` is the name of a trained LoRA. The four shipped with the paper are
`cyberpunk`, `expressionism`, `line-art`, `pixel-art`; add your own by editing
`STYLE_TO_WEIGHTS` in `inference.py`. Optional: `--height` `--width`
`--guidance_scale` `--num_inference_steps` `--seed` (defaults 1024 / 1024 / 3.5
/ 28 / 42).


## Training

One run produces one LoRA for one style. Inputs: a per-style prompts JSON and a
parquet of precomputed T5 embeds.

**JSON manifest** (one record per training pair):

```json
{
  "prompt": "make the man in pixel-art style, keep other area unchanged",
  "image_path": "styled_subdir/styled.png",
  "original_path": "ctx_subdir/src.png",
  "mask_dir": "masks_subdir",
  "image_name": "00001",
  "objects": {
    "man": "pixel-art"
    }
}
```

**Step 1 — embeddings:**

```bash
python training/compute_embeddings.py \
    --json_file     ./prompts.json \
    --local_dataset ./dataset \
    --output_path   ./style.parquet
```

**Step 2 — train one style:**

```bash
DATA_PARQUET=./style.parquet \
JSON_DIR=./prompts \
DATASET_PATH=./dataset \
./training/train.sh --style cyberpunk
```

`train.sh` defaults `JSON_PATH=$JSON_DIR/<style>.json`,
`OUTPUT_DIR=./runs/<style>`, and copies the final weights to
`$LORA_DEST_DIR/<style>.safetensors` (default `./lora_out`). Override any of
those via env var. Re-run with a different `--style` to train another expert.

To plug a freshly-trained LoRA into inference, drop the `.safetensors` next to
the others and add an entry to `STYLE_TO_WEIGHTS` in `inference.py`.
