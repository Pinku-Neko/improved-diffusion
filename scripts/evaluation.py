'''
evaluate the quality of samples
'''
import argparse
from improved_diffusion.script_util import (
    add_dict_to_argparser
)
from improved_diffusion.image_eval import ImageEval

def main():
    import numpy as np
    args = create_argparser().parse_args()
    # create an instance of image_eval
    evaluator = ImageEval(
        batch_size=args.batch_size
        )
    # get image dir from args
    image_real_dir = args.images_real_dir
    image_gen_dir = args.images_gen_dir
    # calculate the FID using evaluator and 2 dir's
    fid_data = evaluator.FID(image_real_dir,image_gen_dir)
    print(f"FID is {fid_data['fid']}, saving result")
    evaluator.save_to_json(fid_data, filename=args.filename)
    import pdb;pdb.set_trace()
                


def create_argparser():
    defaults = dict(
        batch_size = 16,
        images_real_dir = '',
        images_gen_dir = '',
        filename = ''
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
