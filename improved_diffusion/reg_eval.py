'''
evaluate the regression model
'''
from improved_diffusion.image_datasets import _list_image_files_recursively
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import random
# needs fix
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageDataset(th.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img


class Evaluate:
    def __init__(
        self,
        model,
        diffusion,
        data_dir,
        num_samples,
        image_size,
        regression_path=None
    ):
        if th.cuda.is_available():
            self.device = 'cuda'
        self.model = model.to(device=self.device)
        self.regression_path = regression_path
        if self.regression_path is not None:
            self.load_regression()

        self.data_dir = data_dir
        self.diffusion = diffusion
        self.num_samples = num_samples

        self.timesteps = 1000  # TODO fix this later
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # [0,1] to [-1,1]
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def load_data(self):
        # read images from path
        file_paths = _list_image_files_recursively(data_dir=self.data_dir)
        # make dataset
        dataset = ImageDataset(file_paths, self.transform)
        return dataset

    def plot_sample(self):
        # get images from dataset
        dataset = self.load_data()
        self.model.eval()
        # indices for image samples
        indices = random.sample(range(0, len(dataset)), self.num_samples)
        # transform into distributions
        image_tensors = []
        for index in tqdm(indices):
            img = dataset[index]
            image_tensors.append(img)
        image_tensors = th.stack(image_tensors, dim=0)

        # results as nd array
        # 2nd axis: average and std_dev
        predicts = np.zeros((self.timesteps, 2))
        diffs = np.zeros((self.timesteps, 2))
        diffs_abs = np.zeros((self.timesteps, 2))

        for i in tqdm(range(self.timesteps)):
            # time
            time = th.tensor([i]).repeat(self.num_samples)

            # calculate noise
            noise_batch = self.diffusion.q_sample(
                image_tensors, time).to(device=self.device)

            # pass to model
            with th.no_grad():
                predict = self.model(noise_batch)
            predict = (predict * self.timesteps).squeeze().detach().to('cpu')

            # store difference
            difference = predict - time
            difference_abs = abs(difference)

            predicts[i] = np.array([avg_and_std_dev(predict)])
            diffs[i] = np.array([avg_and_std_dev(difference)])
            diffs_abs[i] = np.array([avg_and_std_dev(difference_abs)])

        title_predict = f'Predict of Regression.'
        identity = range(self.timesteps)
        self.plot(plot=plt,title=title_predict, array_1=predicts, array_2=identity)
        plt.show()

        title_error = f'Error of Regression.'
        zeroes = [0]*self.timesteps
        self.plot(plot=plt,title=title_error,array_1=diffs,array_2=zeroes)
        plt.show()

        title_abs_error = f'Absolute Error of Regression.'
        self.plot(plot=plt,title=title_abs_error,array_1=diffs_abs)
        plt.show()

        self.subplot(titles=[title_predict, title_error, title_abs_error],
                     arrays_1= [predicts, diffs, diffs_abs],
                     arrays_2= [identity,zeroes,None])
        
    def plot(self,plot, title, array_1, array_2 = None):
        x_values = range(self.timesteps)
        plot.plot(x_values, array_1[:, 0],
                 label='Average', color='black', zorder=5)
        plot.errorbar(x=x_values, y=array_1[:, 0], yerr=array_1[:, 1],
                     label='Standard Deviation', fmt='o', markersize=0.3, color='blue', zorder=0)
        if array_2 is not None:
            plot.plot(x_values, array_2,color='green')
            # calculate crossings
            crossings = find_crossings(array_2, array_1[:, 0])
            plot.scatter(crossings[:, 0], crossings[:, 1],label = "Crossing", color='red', zorder=10)

        # plot
        if plot is plt:
            plot.xlabel('Timestep')
            plot.ylabel('Error in timestep')
            sample_msg = f' Sample size:{self.num_samples}'
            plot.title(f'{title} {sample_msg}',fontsize=16)
        else:
            plot.set_title(title,fontsize=12)
        plot.legend()
        plot.grid(True)

    def subplot(self, titles, arrays_1, arrays_2):
        fig, axs = plt.subplots(1, len(arrays_1))
        x_values = range(self.timesteps)
        for i in range(len(arrays_1)):
            self.plot(plot=axs[i],title=titles[i],array_1=arrays_1[i],array_2=arrays_2[i])
        fig.suptitle(
            f'Evaluation of Regression Model. Sample size:{self.num_samples}', fontsize=16)
        plt.subplots_adjust(left=0.05, bottom=0.1,
                            right=0.95, top=0.9, wspace=0.3)
        plt.show()

    def load_regression(self):
        # load parameters of regression using self.resume_checkpoint as filename
        print(f'Load parameters from checkpoint {self.regression_path}')
        checkpoint = th.load(
            f"./models/regression/{self.regression_path}", map_location='cpu')['model']
        self.model.load_state_dict(checkpoint)


def find_crossings(array_1, array_2):
    # here forced that 2 arrays have same length
    crossings = []
    for i in range(len(array_1)-1):
        diff = array_1[i]-array_2[i]
        next_diff = array_1[i+1]-array_2[i+1]
        # if the sign of difference changes
        if diff * next_diff < 0:
            crossings.append([i, array_1[i]])
    return np.array(crossings)


def avg_and_std_dev(array):
    average = th.mean(array).item()
    std_dev = th.std(array).item()
    return average, std_dev
