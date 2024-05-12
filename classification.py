import d2l
import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import transforms
from d2l import torch as d2l



class FashionMNIST(d2l.DataModule):
    def __init__(self, resize):
        super.__init__()
        self.save_hyperparameters()
        transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train=torchvision.datasets.FashionMNIST(root=self.root, transform=transform, train=True, download=True)
        self.val=torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=transform, download=True)


data=FashionMNIST((32,32))

@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data=self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)


