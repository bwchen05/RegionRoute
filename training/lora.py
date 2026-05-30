#!/usr/bin/env python

import argparse
import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from transformers import T5TokenizerFast

from loss_utils import cal_attn_loss_by_layer, get_word_idx
from attn_utils import FluxAttnProcessorForTraining, TrainingAttentionStore, register_flux_attention_for_training

if is_wandb_available():
    pass

check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_df_path",
        type=str,
        default=None,
        help=("Path to the parquet file serialized with compute_embeddings.py."),
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help=("Path to the prompts."),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=("Path to training images."),
    )
    parser.add_argument(
        "--enable_attn_loss",
        default=True,
    )
    parser.add_argument(
        "--focus_loss_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--cover_loss_scale",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Used for reading the embeddings. Needs to be the same as used during `compute_embeddings.py`.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora-nf4",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Prodigy", "AdEMAMix"],
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--use_8bit_ademamix",
        action="store_true",
        help="Whether or not to use 8-bit AdEMAMix from bitsandbytes.",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class LoadDataset(Dataset):
    def __init__(
        self,
        data_df_path,
        json_path,
        dataset_path,
        tokenizer,
        size=1024,
        max_sequence_length=77,
    ):
        self.size = size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer

        self.data_df_path = Path(data_df_path)
        if not self.data_df_path.exists():
            raise ValueError("`data_df_path` doesn't exists.")

        self.dataset = self._load_dataset(json_path, dataset_path)

        context_images = [sample["context_image"] for sample in self.dataset]
        instance_images = [sample["image"] for sample in self.dataset]
        self.prompts = [sample["text"] for sample in self.dataset]
        self.mask_list = [sample["mask_list"] for sample in self.dataset]

        image_hashes = [self.generate_image_hash(image) for image in instance_images]

        self.context_images = context_images
        self.instance_images = instance_images
        self.image_hashes = image_hashes

        self.pixel_values = self.apply_image_transformations(instance_images)
        self.context_pixel_values = self.apply_image_transformations(context_images)

        self.data_dict, self.token_positions_dict = self.map_image_hash_embedding(data_df_path=data_df_path)

        self.num_instance_images = len(instance_images)
        self._length = self.num_instance_images

    def _load_dataset(self, json_path, dataset_path):
        import json
        from PIL import Image

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = []

        for i, item in enumerate(data):
            prompt = item["prompt"]
            gt_image_path = item["image_path"]
            ori_image_path = item["original_path"]
            mask_dir = item["mask_dir"]
            image_name = item["image_name"]

            objects = item['objects']

            styled_image_path = os.path.join(dataset_path, gt_image_path)
            ori_image_path = os.path.join(dataset_path, ori_image_path)

            if os.path.exists(styled_image_path) and os.path.exists(ori_image_path):

                styled_image = Image.open(styled_image_path)
                ori_image = Image.open(ori_image_path)

                mask_list = []
                for obj, style in objects.items():
                    mask_path = os.path.join(dataset_path, mask_dir, f"mask_{image_name}_{obj}.png")
                    if not os.path.exists(mask_path):
                        raise ValueError(f"Mask path {mask_path} does not exist")

                    mask_image = Image.open(mask_path)
                    mask_transforms = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    mask_values = mask_transforms(mask_image)
                    if mask_values.shape[0] == 4:
                        mask_values = mask_values[0].unsqueeze(0)

                    if mask_values.max() > 0:
                        mask_values = mask_values / mask_values.max()

                    mask_list.append([
                        style,
                        mask_values
                    ])
                dataset.append({
                    "context_image": ori_image,
                    "image": styled_image,
                    "text": prompt,
                    "mask_list": mask_list,

                })
        return dataset

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = index % self.num_instance_images

        ex = {
            "instance_images": self.pixel_values[idx],
            "context_images": self.context_pixel_values[idx],
            "mask_list": self.mask_list[idx],
        }

        image_hash = self.image_hashes[idx]
        prompt_embeds, pooled_prompt_embeds, text_ids = self.data_dict[image_hash]
        token_positions = self.token_positions_dict[image_hash]

        if isinstance(token_positions, str):
            import json
            token_positions = json.loads(token_positions)

        ex["prompt_embeds"] = prompt_embeds
        ex["pooled_prompt_embeds"] = pooled_prompt_embeds
        ex["text_ids"] = text_ids
        ex["token_positions"] = token_positions

        return ex

    def apply_image_transformations(self, instance_images):
        pixel_values = []

        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            if image.size != (self.size, self.size):
                raise ValueError(f"Image size {image.size} does not match expected size {(self.size, self.size)}")

            image = train_transforms(image)
            pixel_values.append(image)

        return pixel_values

    def convert_to_torch_tensor(self, embeddings: list):
        prompt_embeds = embeddings[0]
        pooled_prompt_embeds = embeddings[1]
        text_ids = embeddings[2]
        prompt_embeds = np.array(prompt_embeds).reshape(self.max_sequence_length, 4096)
        pooled_prompt_embeds = np.array(pooled_prompt_embeds).reshape(768)
        text_ids = np.array(text_ids).reshape(77, 3)
        return torch.from_numpy(prompt_embeds), torch.from_numpy(pooled_prompt_embeds), torch.from_numpy(text_ids)

    def map_image_hash_embedding(self, data_df_path):

        import json

        hashes_df = pd.read_parquet(data_df_path)
        data_dict = {}
        token_positions_dict = {}

        for i, row in hashes_df.iterrows():

            token_pos_str = row['token_positions']
            token_positions = json.loads(token_pos_str)

            embeddings = [row["prompt_embeds"], row["pooled_prompt_embeds"], row["text_ids"]]
            prompt_embeds, pooled_prompt_embeds, text_ids = self.convert_to_torch_tensor(embeddings=embeddings)
            data_dict.update({row["image_hash"]: (prompt_embeds, pooled_prompt_embeds, text_ids)})

            token_positions_dict.update({row["image_hash"]: token_positions})
        return data_dict, token_positions_dict

    def generate_image_hash(self, image):
        return insecure_hashlib.sha256(image.tobytes()).hexdigest()

def collate_fn(examples):
    instance_imgs = torch.stack([ex["instance_images"] for ex in examples]).float()
    context_imgs = torch.stack([ex["context_images"] for ex in examples]).float()

    prompt_embeds = torch.stack([ex["prompt_embeds"] for ex in examples])
    pooled_embeds = torch.stack([ex["pooled_prompt_embeds"] for ex in examples])

    text_ids = torch.stack([ex["text_ids"] for ex in examples])[0]

    mask_lists = [ex["mask_list"] for ex in examples]
    token_positions = [ex["token_positions"] for ex in examples]

    batch = {
        "pixel_values": instance_imgs,
        "context_pixel_values": context_imgs,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_embeds,
        "text_ids": text_ids,
        "mask_list": mask_lists,
        "token_positions": token_positions,
    }

    return batch

def main(args):

    if torch.distributed.is_initialized():
        print(f"Using {torch.distributed.get_world_size()} GPUs")
        print(f"Current rank: {torch.distributed.get_rank()}")
    else:
        print("Single GPU mode")

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

        if accelerator.is_main_process:
            import wandb
            wandb.init(
                project="flux-kontext-attention-control",
                name=f"run_{args.output_dir}_{args.resolution}_{args.rank}",
                config={
                    "learning_rate": args.learning_rate,
                    "batch_size": args.train_batch_size,
                    "resolution": args.resolution,
                    "rank": args.rank,
                    "focus_loss_scale": args.focus_loss_scale,
                    "cover_loss_scale": args.cover_loss_scale,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "max_train_steps": args.max_train_steps,
                    "weighting_scheme": args.weighting_scheme,
                    "optimizer": args.optimizer,
                    "guidance_scale": args.guidance_scale,
                },
                tags=["flux", "kontext", "lora", "attention-control"]
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    bnb_4bit_compute_dtype = torch.float32
    if args.mixed_precision == "fp16":
        bnb_4bit_compute_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        bnb_4bit_compute_dtype = torch.bfloat16

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=nf4_config,
        torch_dtype=bnb_4bit_compute_dtype,
        device_map="auto",
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
            transformer_ = FluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
                quantization_config=nf4_config,
                torch_dtype=bnb_4bit_compute_dtype,
                device_map="auto",
            )
            transformer_ = prepare_model_for_kbit_training(transformer_, use_gradient_checkpointing=False)
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.mixed_precision == "fp16":
            models = [transformer_]
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_ademamix and not args.optimizer.lower() == "ademamix":
        logger.warning(
            f"use_8bit_ademamix is ignored when optimizer is not set to 'AdEMAMix'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    elif args.optimizer.lower() == "ademamix":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use AdEMAMix (or its 8bit variant), please install the bitsandbytes library: `pip install -U bitsandbytes`."
            )
        if args.use_8bit_ademamix:
            optimizer_class = bnb.optim.AdEMAMix8bit
        else:
            optimizer_class = bnb.optim.AdEMAMix

        optimizer = optimizer_class(params_to_optimize)

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    tokenizer = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    train_dataset = LoadDataset(
        data_df_path=args.data_df_path,
        json_path=args.json_path,
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        size=args.resolution,
        max_sequence_length=args.max_sequence_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        ctx_latents_cache = []
        cache_order = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():

                for i in range(len(batch["pixel_values"])):
                    cache_order.append(batch["prompt_embeds"][i].sum().item())

                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                batch["context_pixel_values"] = batch["context_pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )

                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                ctx_latents_cache.append(vae.encode(batch["context_pixel_values"]).latent_dist)
        del vae
        free_memory()

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if args.enable_attn_loss:
        attn_store = TrainingAttentionStore()
        register_flux_attention_for_training(transformer, attn_store)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-dev-lora-nf4"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    focus_loss_scale = args.focus_loss_scale
    cover_loss_scale = args.cover_loss_scale

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    focus_loss_scale = args.focus_loss_scale
    cover_loss_scale = args.cover_loss_scale

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):

            attn_store.reset()
            all_target_indices = set()
            for sample_token_positions in batch["token_positions"]:
                for obj_name, token_indices in sample_token_positions.items():
                    all_target_indices.update(token_indices)

            attn_store.set_target_tokens(list(all_target_indices))

            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                    ctx_latents  = ctx_latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    ctx_pixel_values = batch["context_pixel_values"].to(dtype=vae.dtype, device=accelerator.device)
                    ctx_latents = vae.encode(ctx_pixel_values).latent_dist.sample()
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                ctx_latents = (ctx_latents - vae_config_shift_factor) * vae_config_scaling_factor
                ctx_latents = ctx_latents.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                ctx_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                latent_ids = torch.cat([latent_image_ids, ctx_image_ids], dim=0)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                packed_ctx_latents = FluxKontextPipeline._pack_latents(
                    ctx_latents,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                latent_model_input = torch.cat(
                    [packed_noisy_model_input, packed_ctx_latents], dim=1
                )

                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                text_ids = batch["text_ids"].to(device=accelerator.device, dtype=weight_dtype)

                latent_size = (model_input.shape[2], model_input.shape[3])
                target_img_len = latent_size[0] * latent_size[1] // 4
                text_len = prompt_embeds.shape[1]
                attn_store.set_current_params(target_img_len, text_len)

                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                if args.enable_attn_loss:
                    packed_latent_size = (model_input.shape[2]//2, model_input.shape[3]//2)

                    mask_list = batch["mask_list"]
                    token_positions_batch = batch["token_positions"]
                    batch_size = len(mask_list)
                    all_word_token_idx_ls = []
                    all_gt_seg_ls = []

                    for batch_idx in range(batch_size):
                        sample_mask_list = mask_list[batch_idx]
                        sample_token_positions = token_positions_batch[batch_idx]
                        batch_word_token_idx_ls = []
                        batch_gt_seg_ls = []

                        for mask_item in sample_mask_list:
                            obj_name = mask_item[0]
                            mask_tensor = mask_item[1]
                            if obj_name in sample_token_positions:
                                word_indices = sample_token_positions[obj_name]
                                if word_indices:
                                    batch_word_token_idx_ls.append(word_indices)
                                    batch_gt_seg_ls.append(mask_tensor)
                                else:
                                    print(f"Warning: No token positions found for object '{obj_name}' in batch {batch_idx}")
                            else:
                                print(f"Warning: Object '{obj_name}' not found in precomputed token positions for batch {batch_idx}")

                        all_word_token_idx_ls.append(batch_word_token_idx_ls)
                        all_gt_seg_ls.append(batch_gt_seg_ls)

                    if any(word_token_idx_ls for word_token_idx_ls in all_word_token_idx_ls):
                        total_attn_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                        valid_samples = 0

                        for batch_idx, (word_token_idx_ls, gt_seg_ls) in enumerate(zip(all_word_token_idx_ls, all_gt_seg_ls)):
                            if word_token_idx_ls and gt_seg_ls:
                                loss_dict = cal_attn_loss_by_layer(
                                    attn_store, gt_seg_ls, word_token_idx_ls, packed_latent_size,
                                    tau=0.1, alpha=2.0
                                )

                                torch.cuda.empty_cache()

                                sample_focus_loss = loss_dict["focus_loss"] * focus_loss_scale
                                sample_cover_loss = loss_dict["cover_loss"] * cover_loss_scale
                                sample_attn_loss = sample_focus_loss + sample_cover_loss
                                total_attn_loss = total_attn_loss + sample_attn_loss
                                valid_samples += 1

                                del loss_dict, sample_attn_loss
                                torch.cuda.empty_cache()

                            else:
                                print(f"Warning: No valid token positions found for batch {batch_idx}, skipping attention loss for this sample")

                        if valid_samples > 0:
                            attn_loss = total_attn_loss / valid_samples
                        else:
                            attn_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                    else:
                        print("Warning: No valid token positions found in entire batch, skipping attention loss")
                        attn_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )

                if args.enable_attn_loss:
                    loss = loss + attn_loss

                loss = loss.mean()

                torch.cuda.empty_cache()

                accelerator.backward(loss)

                attn_store.reset()
                torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "train/total_loss": loss.detach().item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/flow_matching_loss": torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean().item(),
            }

            if 'attn_loss' in locals() and attn_loss is not None:
                logs["train/attention_loss"] = attn_loss.item()
                if 'sample_focus_loss' in locals():
                    logs["train/focus_loss"] = sample_focus_loss.item()
                    logs["train/cover_loss"] = sample_cover_loss.item()

            if accelerator.is_main_process and args.report_to == "wandb":
                import wandb
                wandb.log(logs, step=global_step)

            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=None,
        )

        if args.report_to == "wandb":
            import wandb
            wandb.finish()

        accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
