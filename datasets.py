# datasets.py
#
# Utilities for downloading other datasets

from PIL import Image
from torchvision.datasets import VisionDataset

import numpy as np

import os
import requests
import tarfile

# Define CIFAR10C class
class CIFAR10C(VisionDataset):
    _DIR_NAME = "CIFAR-10-C"
    _TAR_NAME = _DIR_NAME+".tar"
    ID = 2535967

    def __init__(self, root, category, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        path = os.path.join(root, self._DIR_NAME, self._DIR_NAME)
        data_path = os.path.join(path, category+".npy")
        label_path = os.path.join(path, "labels.npy")
        if download:
            if os.path.exists(data_path):
                print("already downloaded, skipping...")
            else:
                self._download()
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _download(self):
        resp = requests.get(url=f"https://zenodo.org/api/records/{self.ID}/files/{self._TAR_NAME}/content")
        if resp.status_code != 200:
            raise NotImplementedError
        print("Download finished.")
        path = os.path.join(self.root, self._DIR_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        tar_path = os.path.join(path, self._TAR_NAME)
        with open(tar_path, "wb") as file:
            file.write(resp.content)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=path)

    def __len__(self):
        return len(self.data)

class CIFAR10P(CIFAR10C):
    _DIR_NAME = "CIFAR-10-P"
    _TAR_NAME = _DIR_NAME+".tar"

class CIFAR100C(CIFAR10C):
    _DIR_NAME = "CIFAR-100-C"
    _TAR_NAME = _DIR_NAME+".tar"
    ID = 3555552

