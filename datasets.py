# datasets.py
#
# Utilities for downloading other datasets

from PIL import Image
from torchvision.datasets import VisionDataset
from os.path import join
from zlib import adler32

import numpy as np

import os
import requests
import tarfile

cifar_corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur"
]

# Define CIFAR10C class
class CIFAR10C(VisionDataset):
    _DIR_NAME = "CIFAR-10-C"
    _TAR_NAME = _DIR_NAME+".tar"
    ID = 2535967

    check = {
        "brightness": 2,
        "contrast": 2,
        "defocus_blur": 2,
        "elastic_transform": 2,
        "fog": 2,
        "frost": 2,
        "gaussian_blur": 2,
        "gaussian_noise": 2,
        "glass_blur": 2 ,
        "impulse_noise": 2,
        "jpeg_compression": 2,
        "motion_blur":2 ,
        "pixelate": 2,
        "saturate":2 ,
        "shot_noise":2 ,
        "snow":2 ,
        "spatter":2 ,
        "speckle_noise":2 ,
        "zoom_blur":2,
        "labels": 2
    }


    def __init__(self, root, category, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        path = join(root, self._DIR_NAME, self._TAR_NAME)
        data = join(path, f"{category}.npy")
        labels = join(path, "labels.npy")
        if download and not os.path.exists(data):
            self._download()
        with open(data, "rb") as file:
            content = file.read()
        with open(labels, "rb") as file:
            content_lb = file.read()
        assert self.check[category] == adler32(content)
        assert self.check["labels"] == adler32(content_lb)
        self.data = np.load(content)
        self.labels = np.load(content_lb)


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

