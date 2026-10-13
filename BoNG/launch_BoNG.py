# primary generation script for BoNG (Best-of-N Guidance)
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

from datetime import datetime

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel

import sys
sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

from fks_utils import do_eval

# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data




def main(args):
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps
    if args.bong_t_end is None:
        args.bong_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 1
    if args.use_bong:
        assert args.num_particles > 1
        assert 0 <= args.bong_t_start < args.bong_t_end <= args.num_inference_steps

    # load prompt data # configure pipeline
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt)


    if "mhdang/dpo" in args.model_name and "xl" not in args.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    elif "stabilityai/stable-diffusion-xl-base-1.0" in args.model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    else:
        pipe = FKDStableDiffusion.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # set output directory
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{args.num_inference_steps}_{args.eta}_{args.num_particles}_{args.bong_scale}_{args.bong_t_start}_{args.bong_t_end}_{args.seed}"
    output_dir = os.path.join(args.output_dir, f"{prefix}_{cur_time}")

    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        # make file sleep for a random time
        import time
        print("Sleeping for a random time")
        time.sleep(np.random.randint(1, 10))
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, f"{prefix}_{cur_time}")
        os.makedirs(output_dir, exist_ok=False)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    metrics_to_compute = args.metrics_to_compute.split("#")

    # cache metric fns
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )
    metrics_arr = {
        metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0


    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        fkd_args = dict(
            lmbda=args.lmbda,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_start=args.resample_t_start,
            resampling_t_end=args.resample_t_end,
            guidance_reward_fn=args.guidance_reward_fn,
            potential_type=args.potential_type,
            # BoNG
            use_bong=args.use_bong,
            bong_scale=args.bong_scale,
            bong_t_start=args.bong_t_start,
            bong_t_end=args.bong_t_end,
        )

        start_time = datetime.now()
        images = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            fkd_args=fkd_args,
        )
        end_time = datetime.now()
        time_taken = end_time - start_time

        images = images[0]

        results = do_eval(
            prompt=prompt, images=images, metrics_to_compute=metrics_to_compute
        )

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]

        for metric in metrics_to_compute:
            results[metric]["result"] = [
                results[metric]["result"][i] for i in sorted_idx
            ]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(
                metric,
                metrics_arr[metric]["mean"] / n_samples,
                metrics_arr[metric]["max"] / n_samples,
            )

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")

            os.makedirs(sample_path, exist_ok=True)

            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))

            _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
            if args.num_particles == 1:
                ax = [ax]

            for i, image in enumerate(images):
                ax[i].imshow(image)
                ax[i].axis("off")

            plt.suptitle(prompt[0])
            image_fpath = os.path.join(prompt_path, f"grid.png")
            plt.savefig(image_fpath)
            plt.close()

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)



    # save final metrics
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples
    metrics_arr["time"] = {
        "avg_time_per_prompt": average_time / n_samples,
        "avg_time_per_image": average_time / n_samples / args.num_particles,
    }

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="BoNG_samples")
    parser.add_argument("--save_individual_images", action="store_true")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)

    ## BoNG
    parser.add_argument("--use_bong", action="store_true")
    parser.add_argument("--bong_scale", type=float, default=6.0)
    parser.add_argument("--bong_t_start", type=int, default=40)
    parser.add_argument("--bong_t_end", type=int, default=44,
                        help="exclusive; BoNG is applied on steps [bong_t_start, bong_t_end)")

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#Clip-Score#Clip-Diversity#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt", type=int, default=1000)

    ## FK-Steering baseline
    parser.add_argument("--use_smc", action="store_true")
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=20)
    parser.add_argument("--resample_t_start", type=int, default=20)
    parser.add_argument("--resample_t_end", type=int, default=80)
    parser.add_argument("--potential_type", type=str, default="max")


    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = get_args()
    print(args.seed)
    main(args)
