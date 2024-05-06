from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.transforms = transforms
        import struct
        import gzip
        with gzip.open(image_filename, 'rb') as img_file:
            magic, nums, rows, cols = struct.unpack('>IIII', img_file.read(16))
            self.image = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(nums, rows*cols).astype(np.float32)/255.

        with gzip.open(label_filename, 'rb') as label_file:
            magic, nums = struct.unpack('>II', label_file.read(8))
            self.label = np.frombuffer(label_file.read(), dtype=np.uint8, offset=0).reshape(nums)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        image, label = self.image[index], self.label[index]
        # image = self.apply_transforms(image.reshape(28, 28, 1))
        image = self.apply_transforms(image.reshape((28, 28, -1))).reshape(-1, )
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.label)
        ### END YOUR SOLUTION