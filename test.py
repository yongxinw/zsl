from models import AutoEncoder
from ZSLPrediction import ZSLPrediction
import torch
from dataset import CUBDataset
import numpy as np

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

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()

        self.args = args

        self.model = AutoEncoder(args)
        self.model.load_state_dict(torch.load(args.checkpoint))
        self.model.cuda()
        self.model.eval()

        self.result = {}


        self.train_dataset = CUBDataset(split='train')
        self.test_dataset = CUBDataset(split='test')

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size)

        train_cls = self.train_dataset.get_classes('train')
        test_cls = self.test_dataset.get_classes('test')
        print("Load class")
        print(train_cls)
        print(test_cls)

        self.zsl = ZSLPrediction(train_cls, test_cls)

    # def tSNE(self):



    def conse_prediction(self, mode = 'test'):
        def pred(recon_x, z_tilde, output):
            cls_score = output.detach().cpu().numpy() 
            pred = self.zsl.conse_wordembedding_predict(cls_score, self.args.conse_top_k)
            return pred

        self.get_features(mode = mode, pred_func = pred)

        if (mode+'_pred') in self.result:

            target = self.result[mode + '_label']
            pred = self.result[mode + '_pred']

            acc = np.sum(target == pred)
            total = target.shape[0]
            return acc/total
        else:
            raise NotImplementedError

    def knn_prediction(self, mode = 'test'):
        self.get_features(mode = mode, pred_func = None)

        if (mode+'_feature') in self.result:
            features = self.result[mode+'_feature']
            labels = self.result[mode+'_label']
            self.zsl.construct_nn(features, labels, k = 5, metric = 'cosine', sample_num = 5)
            pred = self.zsl.nn_predict(features)

            acc = np.sum(target == pred)
            total = target.shape[0]

            return acc/total
        else:
            raise NotImplementedError

    def tSNE(self, mode='train'):
        self.get_features(mode= mode, pred_func = None)
        self.zsl.tSNE_visualization(self.result[mode+'_feature'], \
                                    self.result[mode+'_label'], \
                                    mode=mode,
                                    file_name= self.args.tsne_out)



    def get_features(self, mode = 'test', pred_func = None):
        self.model.eval()
        if (mode + '_feature') in self.result:
            print("Use cached result")
            return 


        if mode == 'train':
            loader = self.train_loader
        elif mode == 'test':
            loader = self.test_loader

        all_z = [] 
        all_label = []
        all_pred = []

        for idx, data in enumerate(loader):
            if idx == 3:
                break

            images = Variable(data['image64'].cuda())
            target = Variable(data['class_id'].cuda())

            recon_x, z_tilde, output = self.model(images)
            target = target.detach().cpu().numpy()

            output = F.softmax(output, dim = 1)

            all_label.append(target)
            all_z.append(z_tilde.detach().cpu().numpy())

            if pred_func is not None:
                pred = pred_func(recon_x, z_tilde, output)
                all_pred.append(pred)

        self.result[mode + '_feature'] = np.vstack(all_z) # all features
        # print(all_label)
        self.result[mode + '_label'] = np.hstack(all_label) # all test label

        if pred_func is not None:
            self.result[mode + '_pred'] = np.hstack(all_pred)
            print(self.result[mode + '_pred'].shape)
        print(self.result[mode + '_feature'].shape)
        print(self.result[mode + '_label'].shape)

    







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

    parser.add_argument('--checkpoint', type=str, help='path to checkpoint') 
    parser.add_argument('--conse_top_k', default=10, type=int, help='use top k of conse model')
    parser.add_argument('--tsne_out', default='tSNE.png', type=str, help='tsne output')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)



    tester = Tester(args)
    tester.tSNE(mode = 'test')
    # print('conse acc:' + tester.conse_prediction('test'))
    # print('knn acc:' + tester.knn_prediction('test'))


