import torch as th
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
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
            stop_at = 0.8
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
        self.tolerance=tolerance
        self.batch_size=batch_size
        self.channels=channels
        self.image_size=image_size
        self.stop_at=stop_at

        # for images
        self.image_shape = (batch_size,channels,image_size,image_size)
        self.reverse_transform = transforms.Compose([
            Lambda(lambda t: ((t+1)*127.5).clamp(0,255).to(th.uint8)),
            Lambda(lambda t: t.permute(0,2,3,1))
        ])

    # TODO: make it available with batches
    def forward(self, x=None):
        with th.no_grad():
            # no clue whether there would be a copy issue
            # if no x given, generate a random one
            if x is None:
                x = th.randn(size=self.image_shape,device=self.device)
            # record all repititions
            self.total_rep = 0
            # init t with last timestep (timesteps-1)
            true_t = th.tensor([self.timesteps-1]*x.shape[0],dtype=th.int32,device=self.device)
            threshold = self.timesteps*self.stop_at
            while any(true_t!=0):
                predict_required = th.logical_and(true_t>threshold,true_t != 0)
                predict_not_required = th.logical_and(true_t<=threshold,true_t != 0)
                fast_x = x[predict_required]
                fast_t = true_t[predict_required]
                normal_x = x[predict_not_required]
                normal_t = true_t[predict_not_required]
                if len(fast_t) != 0:
                    x[predict_required],true_t[predict_required] = self.step(fast_x,fast_t)
                if len(normal_t) != 0:
                    with th.no_grad():
                        x[predict_not_required] = self.diffusion.p_sample(self.unet,normal_x,normal_t)['sample']
                        true_t[predict_not_required] -= 1
            # last step, just return p_sample
            last_t = th.zeros(size=(self.image_shape[0],),dtype=th.int32,device=self.device)
            return self.diffusion.p_sample(self.unet,x,last_t)['sample']

    def step(self, x, t):
        with th.no_grad():
            # print current t
            # print(f"predict {t}")
            # sample from diffusion
            # predict the timestep using 
            x_next = self.diffusion.p_sample(self.unet,x,t)['sample']
            t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # repeat for certain number of times
            repitition = 0
            while repitition<self.tolerance:
                # if sampled image has lower predict than true t
                # import pdb;pdb.set_trace()
                if all(t_pred < t):
                    # use the image and use the predict t as true t
                    return x_next, t_pred
                # repeat process until too much
                repitition += 1
                self.total_rep += 1
                # sample from diffusion
                # predict the timestep using regression
                x_next = self.diffusion.p_sample(self.unet,x,t)['sample']
                t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32).squeeze(1)
            # force go down by returning newest x_next
            # one could compare differently generated x_next and return best
            return x_next, t-1


    def sample_images(self, num_samples):
        from tqdm.auto import tqdm
        from datetime import datetime
        import numpy as np
        with th.no_grad():
            batches = []
            # just make enough batches
            num_batches = int(np.ceil(num_samples / self.batch_size))
            progress_bar = tqdm(range(num_batches))
            for i in progress_bar:
                noise = th.randn(self.image_shape,device=self.device)
                batch = self.forward(noise)
                batches.append(batch)
            batches = th.concatenate(batches)
            images = self.reverse_transform(batches[:num_samples]).cpu().numpy()
            now = datetime.now()
            timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
            np.savez(f"./samples/samples_{timestamp}_skip{self.stop_at}_{len(images)}.npz",images)
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
            noise = th.randn(*self.image_shape,device=self.device)
            # fast sample
            fast_sample = self.forward(noise)
            if self.image_shape[0] > 1:
                fast_sample = fast_sample[0].unsqueeze(0)
            fast_sample_image = self.reverse_transform(fast_sample)[0].cpu().numpy()
            # normal sample
            normal_sample = self.diffusion.p_sample_loop(
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

