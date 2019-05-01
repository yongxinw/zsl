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

from sklearn.neighbors import NearestNeighbors

from scipy.misc import imsave

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
        self.val_dataset = CUBDataset(split = 'val')


        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=100,shuffle=True)

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
            print(cls_score)
            pred = self.zsl.conse_wordembedding_predict(cls_score, self.args.conse_top_k)
            return pred

        self.get_features(mode = mode, pred_func = pred)

        if (mode+'_pred') in self.result:

            target = self.result[mode + '_label']
            pred = self.result[mode + '_pred']
            print(target)
            print(pred)
            acc = np.sum(target == pred)
            print(acc)
            total = target.shape[0]
            print(total)
            return acc/float(total)
        else:
            raise NotImplementedError

    def knn_prediction(self, mode = 'test'):
        self.get_features(mode = mode, pred_func = None)

        if (mode+'_feature') in self.result:
            features = self.result[mode+'_feature']
            labels = self.result[mode+'_label']
            print(labels)
            self.zsl.construct_nn(features, labels, k = 5, metric = 'cosine', sample_num = 5)
            pred = self.zsl.nn_predict(features)

            acc = np.sum(labels == pred)
            total = labels.shape[0]

            return acc/float(total)
        else:
            raise NotImplementedError

    def tSNE(self, mode='train'):
        self.get_features(mode= mode, pred_func = None)

        total_num = self.result[mode+'_feature'].shape[0]

        random_index = np.random.permutation(total_num)

        random_index = random_index[:30]

        self.zsl.tSNE_visualization(self.result[mode+'_feature'][random_index,:], \
                                    self.result[mode+'_label'][random_index], \
                                    mode=mode,
                                    file_name= self.args.tsne_out)



    def get_features(self, mode = 'test', pred_func = None):
        self.model.eval()
        if pred_func is None and (mode + '_feature') in self.result:
            print("Use cached result")
            return 
        if pred_func is not None and (mode + '_pred') in self.result:
            print("Use cached result")
            return 


        if mode == 'train':
            loader = self.train_loader
        elif mode == 'test':
            loader = self.test_loader

        all_z = [] 
        all_label = []
        all_pred = []

        for data in tqdm(loader):
            # if idx == 3:
            #     break

            images = Variable(data['image64_crop'].cuda())
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

    def validation_recon(self):
        self.model.eval()
        for idx, data in enumerate(self.val_loader):
            if idx == 1:
                break

            images = Variable(data['image64_crop'].cuda())
            recon_x, z_tilde, output = self.model(images)

            all_recon_images = recon_x.detach().cpu().numpy() #N x 3 x 64 x 64
            all_origi_images = data['image64_crop'].numpy() #N x 3 x 64 x 64

            for i in range(all_recon_images.shape[0]):
                imsave('./recon/recon' + str(i) + '.png', np.transpose(np.squeeze(all_origi_images[i,:,:,:]),[1, 2, 0]))
                imsave('./recon/orig' + str(i) + '.png', np.transpose(np.squeeze(all_recon_images[i,:,:,:]),[1, 2, 0]))



    def test_nn_image(self):
        self.get_features(mode = 'test', pred_func = None)
        self.get_features(mode = 'train', pred_func = None)
        
        N = 100
        random_index = np.random.permutation(self.result['test_feature'].shape[0])[:N]

        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors()
        neigh.fit(self.result['train_feature'])

        test_feature = self.result['test_feature'][random_index,:]
        _, pred_index = neigh.kneighbors(test_feature,1)


        for i in range(N):
            test_index = random_index[i]

            data = self.test_dataset[test_index]
            image = data['image64_crop'].numpy() #1 x 3 x 64 x 64
            print(image.shape)
            imsave('./nn_image/test' + str(i) + '.png', np.transpose(np.squeeze(image),[1, 2, 0]))

            train_index = pred_index[i][0]
            print(train_index)
            data = self.train_dataset[train_index]
            image = data['image64_crop'].numpy() #1 x 3 x 64 x 64
            print(image.shape)
            imsave('./nn_image/train' + str(i) + '.png', np.transpose(np.squeeze(image),[1, 2, 0]))


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
    print("Prediction on test set")
    print('conse acc:')
    print(tester.conse_prediction('test'))
    # print('knn acc:')
    # print(tester.knn_prediction('test'))

    # print("tSNE on train")
    # tester.tSNE(mode = 'train')

    # tester.test_nn_image()  

    tester.validation_recon()  




