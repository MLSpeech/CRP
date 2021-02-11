import torch
from torchvision import datasets, transforms
import torch.optim as optim
from models import models
from configs import DatasetEnum, get_class_count
from models.wide_resnet_28_10 import wide_resnet


def load_check_point(path, model, optimizer):
    global start_epoch
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']+1
    model.eval()
    return model, optimizer, start_epoch

def init(dataset, lr, use_cuda=True):
    normalize_layer = models.NormalizationLayer(dataset)
    num_classes = get_class_count(dataset)

    if dataset == DatasetEnum.MNIST:
        model = models.MNISTModel(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif dataset == DatasetEnum.SVHN:
        lr = 0.002
        model = models.ResNet18(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif dataset == DatasetEnum.CIFAR10:
        lr = 0.02
        model =  wide_resnet(num_classes)
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.Sequential(normalize_layer, model)
    return model, optimizer

def only_test(dataset, batch_size):
    trans = transforms.Compose([transforms.ToTensor()])
    if dataset == DatasetEnum.MNIST:
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=trans),
                                                  batch_size=batch_size, shuffle=False)
    elif dataset == DatasetEnum.CIFAR10:
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data/CIFAR10', train=False, transform=trans),
                                                  batch_size=batch_size, shuffle=False)
    elif dataset == DatasetEnum.SVHN:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN('./data/SVHN', split='test', transform=trans, download=True),
            batch_size=batch_size, shuffle=False)
    return test_loader