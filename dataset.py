from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class CUBDataset(Dataset):
    def __init__(self, root='/media/hdd3/tmp', train=True, transform=None):
        self.root = root
        self.train = train

        self.transform = transform

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        self.classes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ',
                                   names=['class_ids', 'class_names'])
        if self.train:
            train_test_split = image_class_labels[image_class_labels['target'] <= 150]
        else:
            train_test_split = image_class_labels[image_class_labels['target'] > 150]

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

    def __getitem__(self, item):
        data = self.data.iloc[item]
        image_class_id = data['target_x']
        image_path = os.path.join(self.root, 'CUB_200_2011', 'images', data['filepath'])
        edge_path = image_path.replace("images", "edges")
        edge_path = os.path.join(self.root, 'CUB_200_2011', 'edges', edge_path.replace(".jpg", "") + "_edges.jpg")

        image = Image.open(image_path)
        edge = Image.open(edge_path)

        if self.transform is not None:
            image = self.transform(image)

        return image_class_id - 1, image, edge
        pass

    def __len__(self):
        return len(self.data)


def get_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    transform = get_transform()
    dset = CUBDataset(train=True, transform=transform)
    dset[0]
