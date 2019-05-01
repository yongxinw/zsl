import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np

class CUBDataset(Dataset):
    def __init__(self, root='/media/hdd3/tmp', split='train'):
        self.root = root
        self.split = split

        np.random.seed(1024)
        self.transform32_im = self._get_transform(32, im=True)
        self.transform64_im = self._get_transform(64, im=True)

        self.transform32_ed = self._get_transform(32, im=False)
        self.transform64_ed = self._get_transform(64, im=False)

        self._load_metadata()

    def _load_metadata(self):
        bounding_boxes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ',
                                     names=['img_id', 'x', 'y', 'w', 'h'])

        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        self.classes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ',
                                   names=['class_ids', 'class_names'])

        self.classes_train = list(self.classes[self.classes['class_ids'] <= 150]['class_names'])
        self.classes_test = list(self.classes[self.classes['class_ids'] > 150]['class_names'])
        self.classes_train = [x.split('.')[1] for x in self.classes_train]
        self.classes_test = [x.split('.')[1] for x in self.classes_test]
        print(self.classes_train)
        print(self.classes_test)

        if self.split in ['train', 'val', 'all']:
            train_test_split = image_class_labels[image_class_labels['target'] <= 150]
        else:
            train_test_split = image_class_labels[image_class_labels['target'] > 150]

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(bounding_boxes, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split in ['train', 'val']:
            train_val_indices = np.arange(len(self.data))
            inds = np.random.choice(train_val_indices, int(len(train_val_indices) * 0.9), replace=False)
            mask = np.array([i in inds for i in train_val_indices])
            train_indices = train_val_indices[mask]
            val_indices = train_val_indices[~mask]

            if self.split == 'train':
                self.data = self.data.iloc[train_indices]
            elif self.split == 'val':
                self.data = self.data.iloc[val_indices]

    def __getitem__(self, item):
        data = self.data.iloc[item]
        image_class_id = data['target_x']
        image_class_name = self.classes[self.classes['class_ids'] == image_class_id]['class_names'].values[0]

        image_path = os.path.join(self.root, 'CUB_200_2011', 'images', data['filepath'])
        edge_path = image_path.replace("images", "edges")
        edge_path = os.path.join(self.root, 'CUB_200_2011', 'edges', edge_path.replace(".jpg", "") + "_edges.jpg")

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path).convert('RGB')

        iw, ih = image.size
        x, y, w, h = data['x'], data['y'], data['w'], data['h']
        cx, cy = x + w // 2, y + h // 2

        square_size = max(w, h)

        x1, y1 = cx - square_size // 2, cy - square_size // 2
        x1 = np.clip(x1, 0, iw-1)
        y1 = np.clip(y1, 0, ih-1)
        x2, y2 = iw - (x1 + square_size), ih - (y1 + square_size)
        x2 = np.clip(x2, 0, iw-1)
        y2 = np.clip(y2, 0, ih-1)

        image32 = self.transform32_im(image)
        image64 = self.transform64_im(image)
        edge32 = self.transform32_ed(edge)
        edge64 = self.transform64_ed(edge)

        image_crop = image.copy()
        crop = ImageOps.crop(image_crop, border=(x1, y1, x2, y2))

        image32_crop = crop.copy().resize((32, 32))
        image64_crop = crop.copy().resize((64, 64))

        return {"class_id": image_class_id - 1,
                "image32": image32,
                "image64": image64,
                "edge32": edge32,
                "edge64": edge64,
                "image32_crop": TF.to_tensor(image32_crop),
                "image64_crop": TF.to_tensor(image64_crop)
                }

    def _get_transform(self, size, im=True):
        transform_list = list()
        transform_list.append(transforms.Resize((size, size)))
        transform_list.append(transforms.ToTensor())
        # if im:
            # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)

    def get_classes(self, split='test'):
        return self.classes_test if split == 'test' else self.classes_train

    def __len__(self):
        return len(self.data)


def denorm(im):
    std = torch.Tensor([[[0.229]], [[0.224]], [[0.225]]])
    mean = torch.Tensor([[[0.485]], [[0.456]], [[0.406]]])
    im = im * std + mean

    return im


if __name__ == '__main__':
    dset = CUBDataset(split='train')
    ret = dset[7878]
    classid, im32, im64, edge32, edge64 = ret['class_id'], ret['image32'], ret['image64'], ret['edge32'], ret['edge64']

    im32_crop, im64_crop = ret['image32_crop'], ret['image64_crop']

    fig = plt.figure()

    ax2 = fig.add_subplot(1, 3, 1)
    im32 = denorm(im32)
    ax2.imshow(np.transpose(im32, (1, 2, 0)))
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 2)
    ax3.imshow(np.transpose(edge32, (1, 2, 0)))
    ax3.axis('off')

    ax4 = fig.add_subplot(1, 3, 3)
    ax4.imshow(np.transpose(im32_crop, (1, 2, 0)))
    ax4.axis('off')


    ax5 = fig.add_subplot(3, 3, 1)
    im64 = denorm(im64)
    ax5.imshow(np.transpose(im64, (1, 2, 0)))
    ax5.axis('off')

    ax6 = fig.add_subplot(3, 3, 2)
    ax6.imshow(np.transpose(edge64, (1, 2, 0)))
    ax6.axis('off')

    ax7 = fig.add_subplot(3, 3, 3)
    ax7.imshow(np.transpose(im64_crop, (1, 2, 0)))
    ax7.axis('off')

    plt.savefig("test_loader.png")
    # print(classid)
    # print(classname)
    # print(im32.shape)
    # print(im64.shape)
    # print(edge32.shape)
    # print(edge64.shape)
