'''
Evaluate regression model
'''
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    diffusion_and_regression_defaults,
    create_diffusion_regrression,
    args_to_dict,
    add_dict_to_argparser
    )
from improved_diffusion.reg_eval import Evaluate

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    diffusion, regression = create_diffusion_regrression(
        **args_to_dict(args, diffusion_and_regression_defaults().keys())
    )

    print("evaluating...")
    Evaluate(
        diffusion=diffusion,
        model=regression,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        regression_path = args.regression_path,
        clamp=args.clamp
    ).plot_sample()


def create_argparser():
    defaults = dict(
        data_dir="cifar_test",
        image_size=32,
        num_samples=4,
        regression_path = "reg_128_L2_best.pt",
        clamp=False
    )
    defaults.update(diffusion_and_regression_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
