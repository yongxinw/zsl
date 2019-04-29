import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.n_class = args.n_class

        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc1 = nn.Sequential(
            Reshape((-1, self.dim_h * 8 * 4 * 4)),
            nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z) 
            )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            Reshape((-1, self.dim_h * 8)),
            nn.Linear(self.dim_h * 8, 1024),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, self.n_class)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 8 * 8),
            Reshape((-1, self.dim_h * 8, 8, 8)),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 1),
            nn.Sigmoid()
        )



    def forward(self, x):
        feat = self.encoder(x)
        output = self.classifier(feat)
        z = self.fc1(feat)
        recon_x = self.decoder(z)
        return recon_x, z, output

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
    def forward(self, x):
        x = self.main(x)
        return x

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
