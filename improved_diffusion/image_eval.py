import torch
from torchvision.models import inception_v3,Inception_V3_Weights
from torchvision import transforms
import numpy as np
import scipy.linalg
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
            image_real_dir,
            transform = None,
            eps = None
            ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load pre-trained InceptionV3 model
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False, aux_logits=True).to(self.device)
        self.model.eval()
        # Define image transformation
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(299,antialias=True),
            transforms.CenterCrop(299),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming normalization for images
        ])
        self.batch_size: int = batch_size
        self.eps = eps if eps is not None else 1e-6
        # store features of real images
        assert image_real_dir, "invalid directory of real/target images"
        # self.image_real_dir = image_real_dir
        self.images_real = read_images_as_numpy(image_real_dir)
        self.features_real = self.get_features(self.images_real)

    
    def get_features(self,images):
        '''Get features using inception v3 \n
        assume images are npz '''
        from tqdm.auto import tqdm
        dataset = CustomDataset(images,self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(len(dataset),self.batch_size),shuffle=True)
        all_feats = []
        with tqdm(total=len(dataset),desc='Extracting Features') as pbar:
            for data in dataloader:
                with torch.no_grad():
                    data = data.to(self.device)
                    feat = self.model(data).detach().cpu().numpy()
                all_feats.append(feat)
                pbar.update(len(data))
        return np.concatenate(all_feats)
        

    def FID(self,image_gen_dir)->float:
        """Compute the FID between real and generated images"""
        # 2 images are required to be tensor npz in (num_samples, size, size, channels) with arr_0 as key
        images_gen = np.load(image_gen_dir)["arr_0"]
        assert type(self.images_real) == type(images_gen), "type of images mismatch"
        # get features from both
        features_gen = self.get_features(images_gen)
        # calculate mu's and sigma's
        mean_real = np.mean(self.features_real, axis=0)
        cov_real = np.cov(self.features_real, rowvar=False)
        mean_gen = np.mean(features_gen, axis=0)
        cov_gen = np.cov(features_gen, rowvar=False)
        # calculate FID
        fid = np.sum((mean_real - mean_gen) ** 2) + np.trace(cov_real + cov_gen - 2 * scipy.linalg.sqrtm(cov_real.dot(cov_gen)))
        return fid.item()
        
    
    def save_to_self(self, fid: float, image_dir:str)->None:
        files = np.load(image_dir,allow_pickle=True)
        updated_files = {**files, 'fid': fid}
        np.savez(image_dir, **updated_files)

    def plot_fid(self, data_dir: str):
        from matplotlib import pyplot as plt
        import json
        import os
        # Initialize a dictionary to store data by category
        data_by_category = {}

        # Loop through files in the directory
        for filename in os.listdir(data_dir):
            # Check if the file has the correct prefix and is a JSON file
            if filename.startswith("fid_") and filename.endswith(".json"):
                # Extract the numeric part from the filename (e.g., "0.1")
                parts = filename.split("_")
                category = parts[1]
                integer, decimal = parts[2].split(".")[:2]
                x_value = str(float(integer+'.'+decimal)*100)+'%'

                # Load the JSON file
                with open(os.path.join(data_dir, filename), 'r') as file:
                    json_data = json.load(file)

                # Extract the 'key' value from the JSON dictionary
                y_value = json_data.get('fid')

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

def read_images_as_numpy(image_dir):
    from PIL import Image
    from tqdm.auto import tqdm
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    images = []
    for path in tqdm(image_paths,desc='Loading Images into numpy array'):
        image = Image.open(path)
        image_np = np.array(image)
        images.append(image_np)
    return np.array(images)