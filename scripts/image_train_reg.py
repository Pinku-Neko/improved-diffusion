'''
Train encoder+regression on images
'''
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    diffusion_and_regression_defaults,
    create_diffusion_regrression,
    args_to_dict,
    add_dict_to_argparser
)
from improved_diffusion.reg_train import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    diffusion, regression = create_diffusion_regrression(
        **args_to_dict(args, diffusion_and_regression_defaults().keys())
    )
    # logger.log("creating data loader...")
    print("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    # logger.log("training...")
    print("training...")
    TrainLoop(
        diffusion=diffusion,
        regression=regression,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        resume_checkpoint=args.resume_checkpoint,
        clamp=args.clamp,
        loss_type=args.loss_type
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="cifar_train",
        image_size=32,
        lr=1e-4,
        batch_size=1,
        epochs=1000,
        # log_interval=10,
        # save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        clamp=False,
        loss_type='L2'
    )
    defaults.update(diffusion_and_regression_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
