import torch
import os
import re
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args for scale-sampling")

    parser.add_argument('--gamma_1', type=float, default=5.5, help='guidance for denoising process')
    parser.add_argument('--gamma_2', type=float, default=0, help='guidance for inversion process')
    parser.add_argument('--lambda_step', type=int, default=49, help='correction step')
    parser.add_argument('--infer_step', type=int, default=50, help='total inference timestep T')
    parser.add_argument('--image_size', type=int, default=1024, help='The size (height and width) of the generated image.')
    parser.add_argument('--T_max', type=int, default=1, help='Number of rounds for each zigzag iteration step.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to determine the initial latent.')
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--save_dir', type=str, default='./res', help='Path to save the generated images.')
    parser.add_argument(
        '--skip_semantic_proj',
        action='store_true',
        help='Disable semantic-referenced probe-step trajectory reprojection.',
    )
    parser.add_argument(
        '--semantic_k1',
        type=float,
        default=1.0,
        help='Strength factor k1 for semantic-parallel component in probe-step reprojection.',
    )
    parser.add_argument(
        '--semantic_k2',
        type=float,
        default=0,
        help='Strength factor k2 for orthogonal component in probe-step reprojection.',
    )
    parser.add_argument('--log_params_dir', type=str, default=None, help='Directory to save per-step sampling parameters.')

    args = parser.parse_args()
    return args

#TODO user could change this list
prompt_list = ["A happy puppy running.", "A cat to the right of a dog."]

def get_init_latents(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)
    start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
    return start_latents

def sanitize_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9._ -]+', '_', name)
    sanitized = sanitized.strip().replace(' ', '_')
    return sanitized[:120]

if __name__ == '__main__':
    #load args
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    assert args.lambda_step < args.infer_step

    #load model
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained("./models", torch_dtype=dtype,
                                                     variant='fp16',
                                                     safety_checker=None, requires_safety_checker=False, local_files_only=True)
    device = torch.device(args.device)
    pipe = pipe.to(args.device)


    shape = (1, 4, args.image_size // 8, args.image_size // 8)
    #initial latent x_{T}
    init_latent = get_init_latents(args.seed)

    for idx, prompt in enumerate(prompt_list):
        print(f'idx: {idx}\tprompt: {prompt}')
        log_path_base = None
        if args.log_params_dir is not None:
            os.makedirs(args.log_params_dir, exist_ok=True)
            log_path_base = os.path.join(args.log_params_dir, f'{idx}_{sanitize_name(prompt)}')

        #use standard sampling to generate image
        print('start generation via Standard-Sampling...')
        origin_img = pipe(prompt=prompt, shape=shape, guidance_scale = args.gamma_1,
                                num_inference_steps=args.infer_step,latents=init_latent).images[0]
        origin_img.save(os.path.join(args.save_dir,f'{idx}_origin_image_{prompt}.png'))

        #use scale-sampling (semantic projection enabled by default) to generate image
        use_semantic_proj = not args.skip_semantic_proj
        print(f"start generation via SCALE-Sampling... (semantic_proj={'on' if use_semantic_proj else 'off'})")
        scale_sampling_img = pipe.scale_sampling_call(
            prompt=prompt,
            shape=shape,
            guidance_scale=args.gamma_1,
            inv_guidance_scale=args.gamma_2,
            num_inference_steps=args.infer_step,
            latents=init_latent,
            T_max=args.T_max,
            lambda_step=args.lambda_step,
            use_semantic_proj=use_semantic_proj,
            semantic_proj_k1=args.semantic_k1,
            semantic_proj_k2=args.semantic_k2,
            log_params_path=f'{log_path_base}_scale.json' if log_path_base else None,
        ).images[0]
        scale_sampling_img.save(os.path.join(args.save_dir, f'{idx}_scale_sampling_image_{prompt}.png'))

    print('The End!')