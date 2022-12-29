import os
import os.path as path
from typing import Callable, List, Optional

import pandas as pd
import torch
from torch.utils import data


class UCIHAR(data.Dataset):
    """`UCI Human Activity Recognition <https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>`_ dataset. #noqa
    As found in the paper `"Human Activity Recognition Using Smartphones" <https://ieeexplore.ieee.org/document/8567275>`_. #noqa

    Args:
        root (string): Root directory of dataset where the training and testing samples are located.
        train (bool, optional): If True, creates dataset from UCIHAR-training data,
            otherwise from UCIHAR-testing data
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an torch.LongTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes: List[str] = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = path.join(root, "ucihar")
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            print(
                "Download not supported. Please download and extract the file into the root directory"  # noqa
            )

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self._load_data()

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: (sample, target) where target is the index of the target class
        """
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            print("root not found")
            return False

        train_dir = os.path.join(self.root, "train")
        has_train_dir = os.path.isdir(train_dir)
        test_dir = os.path.join(self.root, "test")
        has_test_dir = os.path.isdir(test_dir)

        if not has_train_dir and not has_test_dir:
            print("train or test dir not found")
            return False

        has_train_x = os.path.isfile(os.path.join(train_dir, "X_train.txt"))
        has_train_y = os.path.isfile(os.path.join(train_dir, "y_train.txt"))

        if not has_train_x and not has_train_y:
            print("train x or y file not found")
            return False

        has_test_x = os.path.isfile(os.path.join(test_dir, "X_test.txt"))
        has_test_y = os.path.isfile(os.path.join(test_dir, "y_test.txt"))

        if not has_test_x or not has_test_y:
            print("test x or test y not found")
            return False

        return True

    def _load_data(self):
        data_dir = os.path.join(self.root, "train" if self.train else "test")
        data_file = "X_train.txt" if self.train else "X_test.txt"
        target_file = "y_train.txt" if self.train else "y_test.txt"

        data = pd.read_csv(
            os.path.join(data_dir, data_file), delim_whitespace=True, header=None
        )
        # targets = np.loadtxt(
        #     path.join(data_dir, target_file), delimiter="\n", dtype="int64"
        # ).tolist()
        targets = pd.read_csv(path.join(data_dir, target_file), header=None).to_numpy()

        self.data = torch.tensor(data.values, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.long).squeeze() - 1
