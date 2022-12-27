from pathlib import Path

import torch.nn
import torchvision
from lxml import etree
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class RandomAllImageDataset(Dataset):
    """Dataset class"""

    def __init__(self, images: list[Path], targets: list[Path], split: str):
        self.images_pths = images.copy()
        self.targets_xml = targets.copy()
        self.split = split

        self.targets_xml.sort()
        self.images_pths.sort()

    def __getitem__(self, index) -> tuple[Tensor, tuple[int, int, int, int]]:
        img_pth = self.images_pths[index]
        target_xml = self.targets_xml[index]
        targets = RandomAllImageDataset.parse(target_xml)
        img = torchvision.io.read_image(str(img_pth))
        if self.split == "train":
            tfs = transforms.Compose([
                transforms.Resize((256, 256)),
            ])
            img = tfs(img)
            targets = tuple(float(x / 256) for x in targets)

        return img, targets

    def __len__(self):
        return len(self.targets_xml)

    @staticmethod
    def parse(pth: Path) -> tuple[int, int, int, int]:
        """Given a path to a xml file for this dataset, return a tuple of the bounding box

        Args:
            pth (Path): _description_
        """
        root = etree.fromstring(pth.read_text())
        xmin = int(root.xpath("//xmin")[0].text)
        xmax = int(root.xpath("//xmax")[0].text)
        ymin = int(root.xpath("//ymin")[0].text)
        ymax = int(root.xpath("//ymax")[0].text)
        return xmax, ymax, xmin, ymin
