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
        assert np.int8(np.abs(tolerance)) == tolerance, "invalid value for tolerance, integer between 0 and 127"
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
        # fast sampling
        self.total_fast_steps = 0 # total number of fast sampling steps in one forward
        self.total_fast_steps_list = [] # mean and variance
        self.batch_fast_steps = 0 # total number of batch steps in one forward
        self.batch_fast_steps_list = [] # mean and variance
        # normal sampling
        self.total_normal_steps = 0 # normal sampling
        self.total_normal_steps_list = [] # mean var
        self.batch_normal_steps = 0 # normal sampling
        self.batch_normal_steps_list = []# mean var
        # time used
        self.time = 0 # total time used for sampling
        # repitition
        # consider every sample has record of repitition, size num_samples * timestep
        # init as -1, 0 for successful predict
        self.sample_rep = np.full((num_samples,self.timesteps),self.nan,dtype=np.int8)

    def forward(self, x=None):
        import time
        start = time.time()
        # reset all counters
        self.total_fast_steps = 0
        self.batch_fast_steps = 0
        self.total_normal_steps = 0
        self.batch_normal_steps = 0
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
                    self.batch_fast_steps += 1
                    self.total_fast_steps += len(fast_t)
                if len(normal_t) != 0: # normal sample batch remains
                    x[predict_not_required] = self.sample_fn(self.unet,normal_x,normal_t)['sample']
                    true_t[predict_not_required] -= 1
                    self.batch_normal_steps += 1
                    self.total_normal_steps += len(normal_t)
            # last step, just return sampled
            last_t = th.zeros(size=(self.image_shape[0],),dtype=th.int32,device=self.device)
            self.time = time.time()-start
            return self.sample_fn(self.unet,x,last_t)['sample']

    def step(self, x, t):
        with th.no_grad():
            # sample from diffusion
            # predict the timestep using regression 
            x_next = self.sample_fn(self.unet,x,t)['sample']
            t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # repeat for certain number of times
            repitition = 0
            sample_idx_start = self.current_batch_num*self.batch_size
            current_sample_idx = np.array(range(sample_idx_start,sample_idx_start+x.shape[0]))
            self.sample_rep[current_sample_idx,t.cpu().numpy()] = repitition
            while repitition<self.tolerance:
                # if sampled image has lower predict than true t
                if all(t_pred < t):
                    # use the image and use the predict t as true t
                    return x_next, t_pred
                repitition += 1
                self.sample_rep[current_sample_idx,t.cpu().numpy()] = repitition
                # sample from diffusion
                # predict the timestep using regression
                x_next = self.sample_fn(self.unet,x,t)['sample']
                t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # force go down by returning latest x_next
            # one could compare differently generated x_next and return best
            return x_next, t-1


    def sample_images(self, num_samples: int, model_name: str):
        from tqdm.auto import tqdm
        from datetime import datetime
        
        # counters for model evaluation
        self.init_counters(num_samples=num_samples)
        
        with th.no_grad():
            batches = []
            # make just enough batches, also avaiable for counters
            num_batches = int(np.ceil(num_samples / self.batch_size))
            progress_bar = tqdm(range(num_batches))
            progress_bar.set_description(f"class: {model_name}, cut off at: {self.cut_off}")
            for i in progress_bar:
                self.current_batch_num = i
                noise = th.randn(self.image_shape,device=self.device)
                batch = self.forward(noise)
                batches.append(batch)
                # record counters
                self.total_fast_steps_list.append(self.total_fast_steps)
                self.batch_fast_steps_list.append(self.batch_fast_steps)
                self.total_normal_steps_list.append(self.total_normal_steps)
                self.batch_normal_steps_list.append(self.batch_normal_steps)
                
            batches = th.concatenate(batches)
            images = self.reverse_transform(batches[:num_samples]).cpu().numpy()
            now = datetime.now()
            timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
            # each result is np array of mean and variance
            results = [np.array([np.mean(list),np.var(list)]) for list in [
                self.total_fast_steps_list,
                self.batch_fast_steps_list,
                self.total_normal_steps_list,
                self.batch_normal_steps_list
            ]]
            rep_mean, rep_var = evaluate_2d_array(self.sample_rep)
            np.savez(f"./samples/{model_name}_samples_{timestamp}_skip{self.cut_off}_{len(images)}.npz",
                     arr_0=images,
                     total_fast=results[0],
                     batch_fast=results[1],
                     total_normal=results[2],
                     batch_normal=results[3],
                     rep_mean=rep_mean,
                     rep_var=rep_var,
                     rep_detail=self.sample_rep
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
            # these are used in sample, needs to be defined
            # dont use as template!
            self.current_batch_num = 0
            self.sample_rep = np.zeros((1,self.timesteps))
            # onlt test 1 image
            assert self.batch_size == 1, "current batch size is not 1"
            # define common noise
            # * for unpacking (a,b,c) to a,b,c
            noise = th.randn(*self.image_shape,device=self.device)
            # fast sample
            breakpoint()
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

def evaluate_2d_array(array_2d):
    """
    calculate the mean and variance of a 2d array and
    return numpy array of mean and variance. -1 and 0 if no value
    along 2nd dimension
    """
    breakpoint()
    y_axis_len = array_2d.shape[1]
    means = np.full((y_axis_len,),-1)
    vars = np.full((y_axis_len,),-1)
    for timestep in range(y_axis_len):
        valid_array = array_2d[:,timestep][array_2d[:,timestep]!=-1]
        means[timestep] = np.mean(valid_array) if len(valid_array) != 0 else -1
        vars[timestep] = np.var(valid_array) if len(valid_array) != 0 else 0
    # results = np.full((len(array_2d),2),fill_value=np.nan)
    # for i, list in enumerate(array_2d):
    #     # seems default value of empty list nan
    #     results[i,0] = np.mean(list)
    #     results[i,1] = np.var(list)
    breakpoint()
    return means, vars
    