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

    print("Sampling...")
    Sampler = FastSample(
        unet=unet,
        diffusion=diffusion,
        regression=regression,
        batch_size=args.batch_size,
        timesteps=args.diffusion_steps,
        stop_at=args.stop_at
    )

    # test unet
    # Sampler.test_model()
    Sampler.sample_images(num_samples=args.num_samples,model_name=args.model_name)

    # Sampler.sample_plot()

def create_argparser():
    defaults = dict(
        data_dir="cifar_test",
        regression_path="models/regression/reg_128_L2_best.pt",
        unet_path="models/cifar10_uncond_50M_500K.pt",
        clamp=False,
        batch_size = 16,
        stop_at = 0.8,
        num_samples = 16,
        model_name = 'cifar10'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regression_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
