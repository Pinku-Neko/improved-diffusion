'''
plot the result regarding steps, tol, fid, etc.
'''
import argparse
from improved_diffusion.script_util import (
    add_dict_to_argparser
)

def plot(data_dir: str):
        from matplotlib import pyplot as plt
        import os
        import numpy as np
        from tqdm.auto import tqdm
        # record what cut offs are available
        cut_offs = []
        
        # each category provides an errorbsr plot
        categories_errorbars = ['step_rep']
        dict_err = {}
        
        # each category provides a line plot
        categories_line = ['fast_steps', 'normal_steps']
        dict_line = {}
        
        # each category provides a point of a line plot
        categories_points = ['time', 'fid']
        dict_points = {}
        
        # Loop through files in the directory
        for file in tqdm(os.listdir(data_dir)):
            full_path = f'{data_dir}/{file}'
            with np.load(full_path,allow_pickle=True) as data:
                dict = {}
                cut_off = data['cut_off'].item()
                cut_offs.append(cut_off)
                for category in categories_errorbars:
                    # use errorbar plot
                    list_by_key = sorted(data[category].item().items())
                    result = []
                    for key, values in list_by_key:
                        mean = np.mean(values)
                        var = np.var(values)
                        result.append((key,mean,var))
                    dict = {**dict, category: result}
                dict_err = {**dict_err, cut_off: dict}
                
                dict ={}
                for category in categories_line:
                    dict = {**dict, category: sorted(data[category].item().items())}
                dict_line = {**dict_line, cut_off: dict}
                
                dict ={}
                for category in categories_points:
                    dict = {**dict, category: data[category].item()}
                dict_points = {**dict_points, cut_off: dict}
        
        for dict in [dict_err, dict_line, dict_points]:
            dict = sorted(dict)
        
        # plot dict error bar
        for category in categories_errorbars:
            # ascending cut_off
            sorted_cut_offs = sorted(cut_offs)
            counter = 1
            plt.title(category)
            plt.xlabel('timestep')
            plt.ylabel(f'{category}')
            for cut_off in sorted_cut_offs:
                tuples = dict_err[cut_off][category]
                # tuples contain (timestep, mean, var)
                pos = sorted_cut_offs.index(cut_off)
                offset = -1+2*pos/len(sorted_cut_offs)
                x = [tuple[0] + offset for tuple in tuples]
                y = [tuple[1] for tuple in tuples]
                yerr = [tuple[2]*10 for tuple in tuples]
                # one error bar plot given cut_off
                plt.scatter(x=x,y=y,s=yerr,label=cut_off,alpha=0.3)
                if pos > counter*len(cut_offs)/2 -1 or pos == len(cut_offs)-1:
                    plt.legend()
                    plt.savefig(f'test_{counter}.png')
                    counter += 1
                    plt.close()
        breakpoint()
        for category in categories_line:
            # ascending cut_off
            for cut_off in cut_offs:
                # one line plot given cut_off
                plt.plot()
            plt.legend()
            # save
            plt.savefig()
            
        
        for category in categories_points:
            # ascending cut_off
            collection = []
            for cut_off in cut_offs:
                # one line plot each category
                collection.append()    
            plt.plot(collection)
            plt.legend()
            # save
            plt.savefig()
          
        breakpoint()
        # To sort ascending, use: sorted(thresholds, key=lambda x: x[0])
        # To sort descending, use: sorted(thresholds, key=lambda x: x[0], reverse=True)
        sorted_files = sorted(cut_offs, key=lambda x: x[0])
        for cut_off, dict in sorted_files:
            with np.load(file) as data:
                # Your processing code here
                print(f"Processing {file} with threshold {threshold}")
                # Example of accessing another variable in the .npz file
                # some_array = data['some_array_name']
                # process(some_array)

                # Append values to the category dictionary
                if category not in data_by_category:
                    data_by_category[category] = {'x': [], 'y': []}
                
                data_by_category[category]['x'].append(x_value)
                data_by_category[category]['y'].append(y_value)

        for category, data in data_by_category.items():
            plt.plot(data['x'], data['y'], marker='o', linestyle='-', label=category)

        plt.xlabel('Finish fast sampling at remaining steps')
        plt.ylabel('FID')
        plt.title('FID Evaluation on Fast Sampling')
        plt.legend()
        plt.show()

def main():
    args = create_argparser().parse_args()
    files_dir=args.files_dir
    plot(files_dir)

def create_argparser():
    defaults = dict(
        files_dir = ''
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()