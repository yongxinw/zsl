import argparse

from tqdm import tqdm
import torch
import os
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from utils import log_density_igaussian
from models import AutoEncoder, Discriminator
from dataset import CUBDataset

class Trainer(object):
    def __init__(self, args):
        # load network
        self.G = AutoEncoder(args)
        self.D = Discriminator(args)
        self.G.weight_init()
        self.D.weight_init()
        self.G.cuda()
        self.D.cuda()
        self.criterion = nn.MSELoss()

        # load data
        self.train_dataset = CUBDataset(split='train')
        self.valid_dataset = CUBDataset(split='val')
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=args.batch_size)

        # Optimizers
        self.G_optim = optim.Adam(self.G.parameters(), lr = args.lr_G)
        self.D_optim = optim.Adam(self.D.parameters(), lr = 0.5 * args.lr_D)
        self.G_scheduler = StepLR(self.G_optim, step_size=30, gamma=0.5)
        self.D_scheduler = StepLR(self.D_optim, step_size=30, gamma=0.5)

        # Parameters
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.z_var = args.z_var 
        self.sigma = args.sigma
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

        log_dir = os.path.join(args.log_dir, datetime.now().strftime("%m_%d_%H_%M_%S"))
        # if not os.path.isdir(log_dir):
            # os.makedirs(log_dir)
        self.writter = SummaryWriter(log_dir)

    def train(self):
        global_step = 0
        self.G.train()
        self.D.train()
        ones = Variable(torch.ones(self.batch_size, 1).cuda())
        zeros = Variable(torch.zeros(self.batch_size, 1).cuda())

        for epoch in range(self.epochs):
            self.G_scheduler.step()
            self.D_scheduler.step()
            print("training epoch {}".format(epoch))
            all_num = 0.0
            acc_num = 0.0
            images_index = 0
            for data in tqdm(self.train_loader):
                images = Variable(data['image64'].cuda())
                target_image = Variable(data['image64'].cuda()) 
                target = Variable(data['class_id'].cuda())
                recon_x, z_tilde, output = self.G(images)
                z = Variable((self.sigma*torch.randn(z_tilde.size())).cuda())
                log_p_z = log_density_igaussian(z, self.z_var).view(-1, 1)
                ones = Variable(torch.ones(images.size()[0], 1).cuda())
                zeros = Variable(torch.zeros(images.size()[0], 1).cuda())

                # ======== Train Discriminator ======== #
                D_z = self.D(z)
                D_z_tilde = self.D(z_tilde)
                D_loss = F.binary_cross_entropy_with_logits(D_z+log_p_z, ones) + \
                    F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, zeros)

                total_D_loss = self.lambda_1*D_loss
                self.D_optim.zero_grad()
                total_D_loss.backward(retain_graph=True)
                self.D_optim.step()

                # ======== Train Generator ======== #
                recon_loss = F.mse_loss(recon_x, target_image, reduction='sum').div(self.batch_size)
                G_loss = F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, ones)
                class_loss = F.cross_entropy(output, target)
                total_G_loss = recon_loss + self.lambda_1*G_loss + self.lambda_2*class_loss
                self.G_optim.zero_grad()
                total_G_loss.backward()
                self.G_optim.step()

                # ======== Compute Classification Accuracy ======== #
                values, indices = torch.max(output, 1)
                acc_num += torch.sum((indices == target)).cpu().item()
                all_num += len(target)
                
                # ======== Log by TensorBoardX
                global_step += 1
                if (global_step + 1) % 10 == 0:
                    self.writter.add_scalar('train/recon_loss', recon_loss.cpu().item(), global_step)
                    self.writter.add_scalar('train/G_loss', G_loss.cpu().item(), global_step)
                    self.writter.add_scalar('train/D_loss', D_loss.cpu().item(), global_step)
                    self.writter.add_scalar('train/classify_loss', class_loss.cpu().item(), global_step)
                    self.writter.add_scalar('train/total_G_loss', total_G_loss.cpu().item(), global_step)
                    self.writter.add_scalar('train/acc', acc_num/all_num, global_step)
                    if images_index < 5 and torch.rand(1) < 0.5:
                        self.writter.add_image('train_output_{}'.format(images_index), recon_x[0], global_step)
                        self.writter.add_image('train_target_{}'.format(images_index), target_image[0], global_step)
                        images_index += 1
            if epoch % 2 == 0:
                self.validate(global_step)

    def validate(self, global_step):
        self.G.eval()
        self.D.eval()
        acc_num = 0.0
        all_num = 0.0
        recon_loss = 0.0
        images_index = 0
        for data in tqdm(self.valid_loader):
            images = Variable(data['image64'].cuda())
            target_image = Variable(data['image64'].cuda()) 
            target = Variable(data['class_id'].cuda())
            recon_x, z_tilde, output = self.G(images)
            values, indices = torch.max(output, 1)
            acc_num += torch.sum((indices == target)).cpu().item()
            all_num += len(target)
            recon_loss += F.mse_loss(recon_x, target_image, reduction='sum').cpu().item()
            if images_index < 5:
                self.writter.add_image('valid_output_{}'.format(images_index), recon_x[0], global_step)
                self.writter.add_image('valid_target_{}'.format(images_index), target_image[0], global_step)
                images_index += 1

        self.writter.add_scalar('valid/acc', acc_num/all_num, global_step)
        self.writter.add_scalar('valid/recon_loss', recon_loss/all_num, global_step)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ZRL WAE-GAN')
    parser.add_argument('--epochs', default=250, type=float, help='maximum training epoch')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--z_var', default=2, type=int, help='scalar variance of the isotropic Gaussian prior P(Z)')
    parser.add_argument('--lr_G', default=1e-3, type=float, help='learning rate for AE')
    parser.add_argument('--lr_D', default=4e-4, type=float, help='learning rate for Adversary Network')
    parser.add_argument('--sigma', default=1.41421356, type=float, help='learning rate for Adversary Network')
    parser.add_argument('--lambda_1', default=100, type=float, help='learning rate for Adversary Network')
    parser.add_argument('--lambda_2', default=1000, type=float, help='learning rate for Adversary Network')
    parser.add_argument('--image_dir', default='../images', type=str, help='path to data')
    parser.add_argument('--log_dir', default='log/', type=str, help='path to tensorboard log')
    parser.add_argument('--n_channel', default=3, type=int, help='channel number for image')
    parser.add_argument('--dim_h', default=128, type=int, help='filter numbers')
    parser.add_argument('--n_z', default=64, type=int, help='dimension of the hidden state')
    parser.add_argument('--n_class', default=150, type=int, help='number of classes')
    parser.add_argument('--gpu', default=3, type=int, help='id of gpu')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)

    trainer = Trainer(args)
    trainer.train()
