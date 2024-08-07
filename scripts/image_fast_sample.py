'''
fast sample image using regression
'''
import argparse

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    regression_defaults,
    create_regression,
    args_to_dict,
    add_dict_to_argparser
)
from improved_diffusion.fast_sample import FastSample

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    unet, diffusion= create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    unet.load_state_dict(
        dist_util.load_state_dict(args.unet_path, map_location="cpu")
    )
    
    regression = create_regression(**args_to_dict(args, regression_defaults().keys()))
    regression.load_state_dict(
        dist_util.load_state_dict(args.regression_path, map_location="cpu")['model']
    )
    timesteps = int(args.diffusion_steps if not args.timestep_respacing else args.timestep_respacing)
    print("Sampling...")
    Sampler = FastSample(
        unet=unet,
        diffusion=diffusion,
        regression=regression,
        batch_size=args.batch_size,
        timesteps=timesteps,
        cut_off=args.cut_off,
        tolerance=args.tolerance,
        use_ddim=args.use_ddim
    )

    # test unet
    # Sampler.test_model()
    Sampler.sample_images(num_samples=args.num_samples,model_name=args.model_name)

    # Sampler.sample_plot()

def create_argparser():
    defaults = dict(
        # empty
    )
    preset_cifar10 = dict(
        model_name = 'cifar10',
        regression_path="models/regression/reg_128_L2_best.pt",
        unet_path="models/cifar10_uncond_50M_500K.pt",
        clamp=False,
        # args for unet
        image_size = 32,
        num_channels = 128,
        num_res_blocks = 3,
        learn_sigma = True,
        # args for diffusion
        noise_schedule = 'cosine',
        
        cut_off = 0.8,
        tolerance = 3,
        batch_size = 16,
        num_samples = 16,
        use_ddim = False
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regression_defaults())
    defaults.update(preset_cifar10)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
