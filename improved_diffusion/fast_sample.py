import torch as th
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FastSample:
    def __init__(
            self,
            unet,
            diffusion,
            regression,
            batch_size,
            tolerance = 3,
            channels = 3,
            image_size = 32,
            timesteps = 4000,
            cut_off = 0.8,
            use_ddim=False
            ):
        """
        use 2 models to sample faster
        """
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.unet=unet.to(self.device)
        self.unet.eval()
        self.diffusion=diffusion
        self.regression=regression.to(self.device)
        self.regression.eval()
        self.timesteps=timesteps
        assert int(tolerance) == tolerance, "invalid value for tolerance, integer"
        self.tolerance=tolerance
        self.batch_size=batch_size
        self.channels=channels
        self.image_size=image_size
        self.use_ddim=use_ddim
        self.cut_off=cut_off # at which percentage of total steps fast sampling stops
        self.nan = -1 # use to represent nan for counters
        
        # determine samplers
        self.init_samplers()
        
        # for images
        self.image_shape = (batch_size,channels,image_size,image_size)
        self.reverse_transform = transforms.Compose([
            Lambda(lambda t: ((t+1)*127.5).clamp(0,255).to(th.uint8)),
            Lambda(lambda t: t.permute(0,2,3,1))
        ])
    
    def init_samplers(self):
        if self.use_ddim:
            self.sample_loop = self.diffusion.ddim_sample_loop
            self.sample_fn = self.diffusion.ddim_sample # fn for function
        else:
            self.sample_loop = self.diffusion.p_sample_loop
            self.sample_fn = self.diffusion.p_sample # fn for function
    
    def init_counters(self, num_samples):
        """Initialize the counter variables."""
        from collections import defaultdict
        self.fast_steps = defaultdict(int)
        self.normal_steps = defaultdict(int)
        # time used
        self.time = 0 # total time used for sampling
        # repitition
        self.step_rep = defaultdict(list) # only necessary entry is created

    def forward(self, x=None):
        with th.no_grad():
            # if no x given, generate a random one
            if x is None:
                x = th.randn(size=self.image_shape,device=self.device)
            # init t with last timestep (timesteps-1)
            true_t = th.tensor([self.timesteps-1]*x.shape[0],dtype=th.int32,device=self.device)
            threshold = self.timesteps * self.cut_off
            while any(true_t!=0): # if any time step is not 0 
                predict_required = th.logical_and(true_t>threshold,true_t != 0) # fast sample batch
                predict_not_required = th.logical_and(true_t<=threshold,true_t != 0) # normal sample batch
                # fast sample preprocessing
                fast_x = x[predict_required]
                fast_t = true_t[predict_required]
                # normal sample preprocessing
                normal_x = x[predict_not_required]
                normal_t = true_t[predict_not_required]
                if len(fast_t) != 0: # fast sample batch remains
                    x[predict_required],true_t[predict_required] = self.step(fast_x,fast_t)
                    for timestep in fast_t:
                        self.fast_steps[timestep.item()] += 1
                if len(normal_t) != 0: # normal sample batch remains
                    x[predict_not_required] = self.sample_fn(self.unet,normal_x,normal_t)['sample']
                    true_t[predict_not_required] -= 1
                    for timestep in normal_t:
                        self.normal_steps[timestep.item()] += 1
            # last step, just return sampled
            last_t = th.zeros(size=(self.image_shape[0],),dtype=th.int32,device=self.device)
            return self.sample_fn(self.unet,x,last_t)['sample']

    def step(self, x, t):
        with th.no_grad():
            # sample from diffusion
            # predict the timestep using regression 
            x_next = self.sample_fn(self.unet,x,t)['sample']
            t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # repeat for certain number of times
            repitition = 0
            while repitition<self.tolerance:
                # if sampled image has lower predict than true t
                predict_success = t_pred < t
                if all(predict_success):
                    # use the image and use the predict t as true t
                    for timestep in t:
                        self.step_rep[timestep.item()].append(repitition)
                    return x_next, t_pred
                repitition += 1
                # sample from diffusion
                # predict the timestep using regression
                x_retry = x_next[~predict_success]
                x_retry = self.sample_fn(self.unet,x_retry,t[~predict_success])['sample']
                t_pred[~predict_success] = th.round(self.regression(x_retry) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # force go down by returning latest x_next
            # one could compare differently generated x_next and return best
            for timestep in t:
                self.step_rep[timestep.item()].append(repitition)
            return x_next, t-1


    def sample_images(self, num_samples: int, model_name: str):
        from tqdm.auto import tqdm
        from datetime import datetime
        import time
        start = time.time()
        
        # counters for model evaluation
        self.init_counters(num_samples=num_samples)
        
        with th.no_grad():
            batches = []
            # make just enough batches, also avaiable for counters
            num_batches = int(np.ceil(num_samples / self.batch_size))
            progress_bar = tqdm(range(num_batches))
            progress_bar.set_description(f"class: {model_name}, ddim: {self.use_ddim}, cut off at: {self.cut_off}, tolerance: {self.tolerance}")
            for i in progress_bar:
                noise = th.randn(self.image_shape,device=self.device)
                batch = self.forward(noise)
                batches.append(batch)
                
            batches = th.concatenate(batches)
            images = self.reverse_transform(batches[:num_samples]).cpu().numpy()
            self.time = time.time()-start
            now = datetime.now()
            timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
            # rep_mean, rep_var = evaluate_nested_list(self.step_rep)
            dir = "./samples/ddim" if self.use_ddim else "./samples"
            np.savez(f"{dir}/{model_name}_samples_{timestamp}_cutoff_{self.cut_off}_tol_{self.tolerance}_{len(images)}.npz",
                     arr_0=images,
                     time=self.time,
                     tolerance=self.tolerance,
                     cut_off=self.cut_off,
                     fast_steps=self.fast_steps,
                     normal_steps=self.normal_steps,
                     step_rep=self.step_rep
            )
            print("Sampling complete!")

    def sample_plot(self):
        import matplotlib.pyplot as plt
        with th.no_grad():
            noise = th.randn(*self.image_shape,device=self.device)
            sample = self.forward(noise)
            if self.image_shape[0] > 1:
                sample = sample[0].unsqueeze(0)
            image = self.reverse_transform(sample)[0].cpu().numpy()
            plt.imshow(image)
            plt.show()

    def test_model(self):
        import matplotlib.pyplot as plt
        with th.no_grad():
            # define common noise
            # * for unpacking (a,b,c) to a,b,c
            noise = th.randn(*self.image_shape,device=self.device)
            # fast sample
            fast_sample = self.forward(noise)
            if self.image_shape[0] > 1:
                fast_sample = fast_sample[0].unsqueeze(0)
            fast_sample_image = self.reverse_transform(fast_sample)[0].cpu().numpy()
            # normal sample
            normal_sample = self.sample_loop(
                model=self.unet,
                shape=self.image_shape,
                noise=noise,
                progress=True
            )
            if self.image_shape[0] > 1:
                normal_sample = normal_sample[0].unsqueeze(0)
            normal_sample_image = self.reverse_transform(normal_sample).cpu().numpy().squeeze()
            # subplot
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(fast_sample_image)
            axs[0].set_title(f'fast sample',fontsize=12)
            axs[1].imshow(normal_sample_image)
            axs[1].set_title(f'normal sample',fontsize=12)
            fig.suptitle('comparison')
            plt.show()

def evaluate_nested_list(nested_list):
    """
    calculate the mean and variance of a 2d array and
    return numpy array of mean and variance. -1 and 0 if no value
    along 2nd dimension
    """
    means = {}
    variances = {}
    for index, values in nested_list.items():
        if values:  # Check if the list is not empty
            values_array = np.array(values)
            means[index] = np.mean(values_array)
            variances[index] = np.var(values_array)
        else:
            means[index] = None  # Handle case where list is empty
            variances[index] = None  # Handle case where list is empty
    return means, variances
    