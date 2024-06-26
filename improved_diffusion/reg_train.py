'''
Train regression on images
'''

import torch as th
from torch.optim import AdamW
from tqdm.auto import tqdm


class TrainLoop:
    def __init__(
        self,
        diffusion,
        regression,
        data,
        batch_size,
        lr,
        epochs,
        loss_type,
        clamp,
        resume_checkpoint
    ):
        self.diffusion = diffusion
        self.timesteps = 1000  # TODO fix this later
        self.regression = regression
        self.regression_dim = 128  # TODO fix this later
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.clamp = clamp
        self.loss_type = loss_type
        self.crit = self.setup_optimizer()

        self.model_params = list(self.regression.parameters())
        self.opt = AdamW(params=self.model_params, lr=self.lr)
        
        
        self.losses = []
        self.cur_loss = 0.0

        self.step = 0
        if th.cuda.is_available():
            self.device = 'cuda'

        self.regression = self.regression.to(self.device)
        self.resume_checkpoint = resume_checkpoint
        if self.resume_checkpoint:
            self.load_regression()

    def get_avg_loss(self):
        return (sum(self.losses)/len(self.losses))

    def run_loop(self):
        p_bar = tqdm(range(
            self.epochs), desc=f"Regression lr {self.lr}; loss: {round(self.cur_loss,6)}")
        for epoch in p_bar:
            # take step
            self.step += 1
            batch = next(self.data)[0].to(self.device)  # bad coding
            self.train_step(batch=batch)
            # test set
            # save model
            if epoch % 100 == 0:
                self.save_regression()
            # update progress bar
            p_bar.set_description(
                f"Regression lr {self.lr}; loss: {self.get_avg_loss()}")

    def train_step(self, batch):
        # make random t samples
        batch_size = len(batch)
        t_samples = th.randint(
            0, self.timesteps, (batch_size, 1), device=self.device)
        # make q samples
        q_samples = self.diffusion.q_sample(batch, t_samples)
        # clamp the samples between [-1,1]
        if self.clamp:
            q_samples = th.clamp(q_samples, -1, 1)
        # step on random timesteps and model with q samples as input
        predicts = self.regression(q_samples)
        # compute loss
        # normalize output
        t_samples_norm = t_samples / self.timesteps
        loss = self.crit(predicts, t_samples_norm)
        # backpropagation
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # save losses
        self.cur_loss = loss.item()
        self.losses.append(self.cur_loss)

    def setup_optimizer(self):
        if self.loss_type == 'L1':
            crit = th.nn.L1Loss()
        elif self.loss_type == 'L2':
            crit = th.nn.MSELoss()
        else:
            raise ValueError('Invalid loss type')
        print(f"{self.loss_type} Loss")
        return crit
    
    def save_regression(self):
        clamp = "_clamp" if self.clamp else ""
        th.save({"model": self.regression.state_dict(), "optimizer": self.opt.state_dict()},
                f"./models/regression/reg_{self.regression_dim}_{self.loss_type}_best{clamp}.pt")  # fix later

    def load_regression(self):
        # load parameters of regression using self.resume_checkpoint as filename
        print(f'Resuming from checkpoint {self.resume_checkpoint}')
        checkpoint = th.load(
            f"./models/regression/{self.resume_checkpoint}", map_location='cpu')['model']
        self.regression.load_state_dict(checkpoint)
