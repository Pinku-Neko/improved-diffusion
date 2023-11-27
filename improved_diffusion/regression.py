from torch import nn
from improved_diffusion.script_util import create_model
from improved_diffusion.unet import UNetModel

class Regression(nn.Module):
    '''
    ordinary regression model with given input and layer dim \n
    1 hidden layer \n
    all linear fully connected with relu \n
    -input_dim: dimension of input layer \n
    -hidden_dim: dimension of 1 hidden layer \n
    -return: model, which outputs a value
    '''

    def __init__(self, input_dim, hidden_dim):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        predict = self.fc2(x)
        return predict

embedding_dim_img_32 = 1*256*4*4
class Advanced_Regression(nn.Module):
    '''
    a model consisting of trainable unet and regression \n
    -layer_dim: the dimension of hidden layer in part regression
    -return: a model, which outputs a value
    '''
    # now we know torch.Size([1, 256, 4, 4]) is the output after encoder of unet
    def __init__(self, hidden_dim):
        super().__init__()
        self.unet = create_model(
        # default for cifar-10 unconditional
        image_size=32,
        num_channels=128,
        num_res_blocks=2,
        learn_sigma=True,
        dropout=0.3,
        # from here on default
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True
    )
        self.reg = Regression(input_dim=embedding_dim_img_32, hidden_dim=hidden_dim)

    def forward(self, x):
        embedding = self.unet(x)
        flat_emb = embedding.view(embedding.size(0), -1)
        predict = self.reg(flat_emb)
        return predict
    