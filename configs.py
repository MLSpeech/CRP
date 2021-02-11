import enum

class DatasetEnum(enum.Enum):
    MNIST = 0
    CIFAR10 = 1
    SVHN = 2

class AttackType(enum.Enum):
    FGSM_ATTACK = 0
    PGD_ATTACK = 1
    BIM = 2
    MIM = 3
    CW = 4

def get_attack_params(dataset):
    epsilon = 0.3 if dataset == DatasetEnum.MNIST else round(8./255,2)
    alpha = 0.01 if dataset == DatasetEnum.MNIST else 2. / 255
    return epsilon, alpha

def get_class_count(dataset):
    if dataset == DatasetEnum.MNIST or dataset == DatasetEnum.SVHN or dataset == DatasetEnum.CIFAR10:
        return 10

def get_dataset_mean_std(dataset):
    if dataset == DatasetEnum.CIFAR10:
        mean = [0.4914,0.4822,0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == DatasetEnum.MNIST:
        mean = [0.1307]
        std = [0.3081]
    elif dataset == DatasetEnum.SVHN:
        mean = [0.4376821, 0.4437697, 0.47280442]
        std = [0.19803012, 0.20101562, 0.19703614]
    return mean, std

def get_dataset_default_lr(dataset):
    lr = -1
    if dataset == DatasetEnum.CIFAR10:
        lr = 0.02
    elif dataset == DatasetEnum.MNIST:
        lr = 0.001
    elif dataset == DatasetEnum.SVHN:
        lr = 0.002
    return lr

def get_dataset_decay(dataset):
    decay = []
    if dataset == DatasetEnum.CIFAR10:
        decay = [260, 320, 380, 400, 450, 500]
    elif dataset == DatasetEnum.MNIST:
        decay = [40]
    elif dataset == DatasetEnum.SVHN:
        decay = [120, 220,280,320,350]
    return decay