import torch
from torchvision.models import inception_v3
from torchvision import transforms
import numpy as np
import scipy.linalg

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image

class ImageEval:
    def __init__(
            self,
            batch_size,
            transform = None,
            eps = None
            ):
        # Load pre-trained InceptionV3 model
        self.model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        self.model.eval()
        # Define image transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming normalization for images
        ])
        self.batch_size: int = batch_size
        self.eps = eps if eps is not None else 1e-6
    
    def get_features(self,images):
        '''Get features using inception v3 \n
        assume images are npz '''
        from tqdm.auto import tqdm
        dataset = CustomDataset(images,self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(len(dataset),self.batch_size),shuffle=True)
        all_feats = []
        with tqdm(total=len(dataloader),desc='Extracting Features') as pbar:
            for data in dataloader:
                with torch.no_grad():
                    feat = self.model(data).detach().cpu().numpy()
                all_feats.append(feat)
                pbar.update(len(data))
        return np.concatenate(all_feats)
        

    def FID(self,image_real_dir,image_gen_dir)->dict:
        """Compute the FID between real and generated images"""
        # 2 images are required to be tensor npz in (num_samples, size, size, channels) with arr_0 as key
        images_real = np.load(image_real_dir)["arr_0"]
        images_gen = np.load(image_gen_dir)["arr_0"]
        assert type(images_real) == type(images_gen), "type of images mismatch"
        # get features from both
        features_real = self.get_features(images_real)
        features_gen = self.get_features(images_gen)
        # calculate mu's and sigma's
        mean_real = np.mean(features_real, axis=0)
        cov_real = np.cov(features_real, rowvar=False)
        mean_gen = np.mean(features_gen, axis=0)
        cov_gen = np.cov(features_gen, rowvar=False)
        # calculate FID
        fid = np.sum((mean_real - mean_gen) ** 2) + np.trace(cov_real + cov_gen - 2 * scipy.linalg.sqrtm(cov_real.dot(cov_gen)) + self.eps)
        fid_data: dict = {
            'fid': fid,
            'mu_real': mean_real,
            'sigma_real': cov_real,
            'mu_generated': mean_gen,
            'sigma_generated': cov_gen
        }
        return fid_data
    
    def save_to_json(self, fid_data: dict, indent: int =None):
        import json
        indent = 4 if indent is None else indent
        with open('fid_data.json', 'w') as file:
            json.dump(fid_data, file, indent=indent)

    def plot_fid(self, list_fid_data: list[dict]):
        import matplotlib.pyplot as plt
        #TODO: draw line plot use fids from bad to good
        pass

