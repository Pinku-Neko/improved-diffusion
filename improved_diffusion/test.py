from image_datasets import _list_image_files_recursively
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch as th
from mpi4py import MPI
import blobfile as bf
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img


data_dir="cifar_train"
# all_paths = _list_image_files_recursively(data_dir=data_dir)
# img = mpimg.imread(all_paths[0])
noise = th.randn((32,32))
all_paths = _list_image_files_recursively(data_dir)
transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.permute(1,2,0)),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
# tensors = th.stack([transform(Image.open(file_path)) for file_path in all_paths])
dataset = CustomDataset(all_paths,transform)
# transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Lambda(lambda t: t.permute(1,2,0))])
import pdb;pdb.set_trace()
# transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Lambda(lambda t: t.permute(1,2,0))])
# tensors = th.stack([transform(img) for img in dataset])