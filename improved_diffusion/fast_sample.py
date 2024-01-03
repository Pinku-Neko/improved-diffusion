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
        self.stop_at=stop_at

        # for images
        self.image_shape = (batch_size,channels,image_size,image_size)
        self.reverse_transform = transforms.Compose([
            Lambda(lambda t: ((t+1)*127.5).clamp(0,255).to(th.uint8)),
            Lambda(lambda t: t.permute(0,2,3,1))
        ])

    # TODO: make it available with batches
    def forward(self, x=None):
        # no clue whether there would be a copy issue
        # if no x given, generate a random one
        if x is None:
            x = th.randn(size=self.image_shape,device=self.device)
        # record all repititions
        self.total_rep = 0
        # init t with last timestep (timesteps-1)
        self.true_t = self.timesteps-1
        while self.true_t != 0:
            if self.true_t >= self.timesteps*self.stop_at:
                x = self.step(x)
            else:
                with th.no_grad():
                    t = th.tensor([self.true_t],device=self.device)
                    x = self.diffusion.p_sample(self.unet,x,t)['sample']
                    self.true_t -= 1
                    if self.true_t % 100 == 0:
                        print(self.true_t)
        # last step, just return p_sample
        last_t = th.zeros(size=(self.image_shape[0],),dtype=th.int32,device=self.device)
        return self.diffusion.p_sample(self.unet,x,last_t)['sample']

    def step(self, x):
        # print current t
        print(f"timestep {self.true_t}")
        # sample from diffusion
        current_t = th.tensor([self.true_t],device=self.device)
        # predict the timestep using 
        with th.no_grad():
            x_next = self.diffusion.p_sample(self.unet,x,current_t)['sample']
            t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32)
        # repeat for certain number of times
        repitition = 0
        while repitition<self.tolerance:
            # if sampled image has lower predict than true t
            if t_pred.item() < self.true_t:
                # use the image and use the predict t as true t
                self.true_t = t_pred.item()
                return x_next
            # repeat process until too much
            repitition += 1
            self.total_rep += 1
            # sample from diffusion
            current_t = th.tensor([self.true_t],device=self.device)
            # predict the timestep using regression
            with th.no_grad():
                x_next = self.diffusion.p_sample(self.unet,x,current_t)['sample']
                t_pred = th.round(self.regression(x_next) * self.timesteps).to(dtype=th.int32)
        # force go down by returning newest x_next
        # one could compare differently generated x_next and return best
        self.true_t -= 1
        return x_next

    def sample_images(self, num_samples):
        from tqdm.auto import tqdm
        batches = []
        # just make enough batches
        num_batches = num_samples // self.batch_size + 1
        for i in tqdm(range(num_batches)):
            noise = th.randn(self.image_shape)
            batch = self.forward(noise)
            batches.append(batch)
        batched = th.concatenate()


    def sample_plot(self):
        import matplotlib.pyplot as plt
        noise = th.randn(*self.image_shape,device=self.device)
        sample = self.forward(noise)
        image = self.reverse_transform(sample)[0].cpu().numpy()
        plt.imshow(image)
        plt.show()

    def test_model(self):
        import matplotlib.pyplot as plt
        # define common noise
        noise = th.randn(*self.image_shape,device=self.device)
        # fast sample
        fast_sample = self.forward(noise)
        fast_sample_image = self.reverse_transform(fast_sample)[0].cpu().numpy()
        # normal sample
        normal_sample = self.diffusion.p_sample_loop(
            model=self.unet,
            shape=self.image_shape,
            noise=noise,
            progress=True
        )
        normal_sample_image = self.reverse_transform(normal_sample).cpu().numpy().squeeze()
        # subplot
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(fast_sample_image)
        axs[0].set_title(f'fast sample',fontsize=12)
        axs[1].imshow(normal_sample_image)
        axs[1].set_title(f'normal sample',fontsize=12)
        fig.suptitle('comparison')
        plt.show()

