'''
evaluate the quality of samples
'''
import argparse
import os
from improved_diffusion.script_util import (
    add_dict_to_argparser
)
from improved_diffusion.image_eval import ImageEval

def main():
    args = create_argparser().parse_args()
    # get image dir from args
    image_real_dir = args.images_real_dir
    image_gen_dir = args.images_gen_dir
    gen_files = os.listdir(image_gen_dir)
    # create an instance of image_eval
    evaluator = ImageEval(
        batch_size=args.batch_size,
        image_real_dir=image_real_dir
        )
    # calculate the FID using evaluator and 2 dir's
    for gen_file in gen_files:
        match args.job:
            case 'fid':
                full_path = f'{image_gen_dir}/{gen_file}'
                fid = evaluator.FID(full_path)
                evaluator.save_to_self(fid=fid,image_dir=full_path)
                print(f"FID is {fid}, result saved")
            case 'plot':
                evaluator.plot_fid(data_dir='fid_scores')
            case _:
                print("invalid job requested")
                


def create_argparser():
    defaults = dict(
        batch_size = 128,
        images_real_dir = '',
        images_gen_dir = '',
        job = ''
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
