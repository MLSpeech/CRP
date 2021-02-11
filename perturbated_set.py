import numpy as np
import os
import torch
import math
import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms as trans
import torchvision.transforms.functional as F
import os

class MNISTPertubatedSetLoader(datasets.MNIST):
    def __init__(self, folder_path, pert_count, pert_range, pert_file="", transforms=None, train=True):
        if not os.path.isfile(f"{folder_path}/processed/training.pt"):
            datasets.MNIST('./data', train=True, transform=trans, download=True)
        else:
            self.data, self.labels = torch.load(f"{folder_path}/processed/training.pt")
        self.train=True
        self.transform = transforms
        self.transform_image = True
        if train:
            self.orig_train_size = self.data.shape[0]
            self.ex_count = len(self.data)
            self.pert_count = pert_count
            if pert_file == "":
                seed = int(os.environ['seed'])
                np.random.seed(seed)
                self.perturbations = torch.FloatTensor(
                    np.random.uniform(-1, 1, (self.ex_count * self.pert_count, 28, 28)) * pert_range)
            else:
                self.perturbations = torch.FloatTensor(np.load(pert_file)) * pert_range
            self.load_pert_with_item = True
            self.load_pert_with_item = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) where target is class_index of the target class.
        """
        image, label = self.data[int(index % len(self.data))], self.labels[int(index % len(self.data))]
        image = self.transform(Image.fromarray(image.numpy(), mode='L'))
        if index >= self.orig_train_size:
            pert_index = index - self.orig_train_size
            image = image + self.perturbations[pert_index]
            image = torch.clamp(image, 0., 1.)

        return image, label

    def __len__(self):
        return self.data.shape[0] * (self.pert_count + 1)


class SVHNPertubatedSetLoader(datasets.SVHN):
    def __init__(self, root, split = 'train', transform = None, download = False, pert_count = 0, pert_range=8./255):
        datasets.SVHN.__init__(self, root, split, transform=transform, download=download)
        self.pert_count = pert_count
        self.ex_count = len(self.data)
        self.pert_range = pert_range
        seed = int(os.environ['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.perturbations = torch.FloatTensor(np.random.uniform(-1,1,(self.ex_count * self.pert_count, 3, 32, 32))) * pert_range
        self.load_from_base = True
        self.preprocess_images_with_pert()

    def preprocess_images_with_pert(self, seed=134):
        trans_images = []
        torch.manual_seed(seed)
        np.random.seed(seed)
        for i in range(len(self.data)):
            trans_images.append(datasets.SVHN.__getitem__(self,i))
        self.images = trans_images
        self.load_from_base = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label, perturbations, pert. labels) where target is class_index of the target class.
        """
        if self.load_from_base:
            image, label = datasets.SVHN.__getitem__(self,int(index%len(self.data)))
        else:
            image, label = self.images[int(index%len(self.data))]

        if index >= self.ex_count:
            pert_index = index - self.ex_count
            image = image + self.perturbations[pert_index]
            image = torch.clamp(image,0.,1.)
        return image, label

    def __len__(self):
        return len(self.data)*(self.pert_count+1)


class CIFAR10PertubatedSetLoader(datasets.CIFAR10):
    def __init__(self, root, pert_range, pert_count, transforms=None, train=True, download=True, pert_file=""):
        datasets.CIFAR10.__init__(self, root, train=train, download=download, transform=transforms)

        if train:
            self.orig_train_size = self.train_data.shape[0]
            self.pert_count = pert_count
            self.ex_count = len(self.train_data) if train else len(self.test_data)
            self.pert_range = pert_range
            pert_path = pert_file
            if pert_file == "":
                seed = int(os.environ['seed'])
                np.random.seed(seed)
                self.perturbations = torch.FloatTensor(
                    np.random.uniform(-1, 1, (self.ex_count * self.pert_count, 3, 32, 32)) * pert_range)
            else:
                self.perturbations = np.reshape(np.load(pert_path)[:self.ex_count * self.pert_count],
                                                (self.ex_count * self.pert_count, 3, 32, 32)) * pert_range
            self.preprocess_images_with_pert()

    def preprocess_images_with_pert(self):
        images = np.zeros((self.train_data.shape[0], 3, 32, 32))
        for index, image in enumerate(self.train_data):
            images[index] = self.transform(Image.fromarray(image))
        self.train_data = torch.FloatTensor(images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) where target is class_index of the target class.
        """
        if self.train:
            image, label = self.train_data[int(index % len(self.train_data))], self.train_labels[
                int(index % len(self.train_data))]
            image = torch.FloatTensor(image)
            if index >= self.orig_train_size:
                pert_index = index - self.orig_train_size
                image = image + torch.FloatTensor(self.perturbations[pert_index])
                image = torch.clamp(image, 0., 1.)
        else:
            image, label = self.test_data[index], self.test_labels[index]

        return image, label

    def __len__(self):
        return self.train_data.shape[0] * (self.pert_count + 1)
