import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from train import train, test, test_batch
import os
from attacks.attacks import FGSMAttack, invoke_pgd_attack, fgsm_attack
import numpy as np
from perturbated_set import MNISTPertubatedSetLoader, SVHNPertubatedSetLoader, CIFAR10PertubatedSetLoader
from configs import DatasetEnum, AttackType, get_dataset_default_lr, get_attack_params, get_dataset_decay
import utils


def load_data(dataset, pert_file, pert_count, train_batch_size, pert_range):
    trans = transforms.Compose([transforms.ToTensor()])
    if dataset == DatasetEnum.MNIST:
        train_dataset = MNISTPertubatedSetLoader("./data/MNIST", pert_count,pert_range, pert_file,transforms=trans, train=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', train=False, transform=trans, download=True),
                                                  batch_size=train_batch_size, shuffle=False)
    if dataset == DatasetEnum.SVHN:
        crop_tran = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        norm_tran = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trans = crop_tran
        train_dataset = SVHNPertubatedSetLoader('./data/SVHN',download=True,pert_range=pert_range, pert_count=pert_count, transform=trans)
        test_loader = torch.utils.data.DataLoader(datasets.SVHN('./data/SVHN', split='test', transform=norm_tran, download=True),
                                                batch_size=128, shuffle=False,pin_memory=True)
    elif dataset == DatasetEnum.CIFAR10:
        cifar_trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        train_dataset = CIFAR10PertubatedSetLoader('./data/CIFAR10', pert_range, pert_count,transforms=cifar_trans, train=True, download=True,pert_file=pert_file)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data/CIFAR10', train=False, transform=trans),
                                                  batch_size=128, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    return train_loader, test_loader


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(train_loader, test_loader, use_cuda, epochs, model, optimizer, dataset, loss):
    best_acc = -np.inf
    acc_in_ep = []

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, use_cuda, verbose=True)
        attack_acc = best_acc

        decays = get_dataset_decay(dataset)
        if epoch in decays:
            if dataset == DatasetEnum.MNIST:
                adjust_lr(optimizer, optimizer.param_groups[0]['lr'] * 0.2)
        elif dataset == DatasetEnum.SVHN:
            if epoch == 100:
                adjust_lr(optimizer, optimizer.param_group['lr'] / 2)
            else:
                adjust_lr(optimizer, optimizer.param_groups[0]['lr'] / 10)
        elif dataset == DatasetEnum.CIFAR10:
            if epoch == 100:
                adjust_lr(optimizer, optimizer.param_group['lr'] / 2)
            elif epoch == 200:
                adjust_lr(optimizer, 0.004)
            else:
                adjust_lr(optimizer, optimizer.param_groups[0]['lr'] * 0.2)


        # Test Robustness & Gen.
        if epoch in range(10, epochs, 10):
            print("evaluating test in chkpt " + checkpoint_path)
            if dataset == DatasetEnum.MNIST:
                ep1, ep2 = 0.3, 0.1
            elif dataset == DatasetEnum.CIFAR10:
                ep1, ep2 = 0.01, 8 / 255.
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            fgsm_acc1, fgsm_acc1_val = fgsm_attack(test_loader, model, use_cuda, loss, test_batch, init_ep=ep1,
                                                   max_range=1)
            fgsm_acc2, fgsm_acc2_val = fgsm_attack(test_loader, model, use_cuda, loss, test_batch, init_ep=ep2,
                                                   max_range=1)
            test_loss, test_acc = test(test_loader, model, use_cuda, dataset)
            acc_in_ep.append((test_loss, test_acc, fgsm_acc1, fgsm_acc2))
            attack_acc = fgsm_acc2_val[0][0]
        if attack_acc > best_acc:
            print('Saving model...')
            best_acc = attack_acc
            state = {
                'net': model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict()
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + checkpoint_path)
    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':

    # Training settings
    seed = np.random.randint(1,600)
    parser = argparse.ArgumentParser(
        description='Adversarial Training')
    parser.add_argument('--dataset', type=int, default=1,
                        help='which dataset to use, 0 for MNIST, 1 for CIFAR10, 2 for SVHN')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='training batch size')
    parser.add_argument('--lr', type=int, default=-1, metavar='N',
                        help='initial learning rate - if not specified uses 0.02 for cifar and 0.001 for mnist')
    parser.add_argument('--pert_count', type=int, default=10,
                        metavar='N', help='number of perturbations')
    parser.add_argument('--pert_range', type=float, default=8/255,
                        metavar='N', help='perturbations max value range')
    parser.add_argument('--model_path', default="",
                        metavar='N', help='path to trained model')
    parser.add_argument('--chkpt_path', default="default_path   ",
                        metavar='N', help='name of checkpoint path in which model will be saved')
    parser.add_argument('--epochs', type=int, default=300,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--pert_file', default=None,
                        help='npy file with perturbations')
    parser.add_argument('--attack', type=int, default=0, metavar='N',
                        help='which attack to use - 0 for fgsm, 1 for pgd')
    parser.add_argument('--train_type', type=int, default=0, metavar='N',
                        help='what type of perturbations to use in training - 0 for no perturbations, 1 for distribution based, 2 for attack based')
    parser.add_argument('--seed', type=int, default=seed,
                        metavar='N')
    parser.add_argument('--pgd_iters', type=int, default=100,
                        metavar='N')
    parser.add_argument('--epsilon', type=float, default=None,
                        metavar='N')
    parser.add_argument('--step_size', type=float, default=None,
                        metavar='N')
    args = parser.parse_args()

    if args.pert_range is None:
        Exception("must specify perturbation range")

    batch_size = args.batch_size if args.batch_size is None else 128
    checkpoint_path = args.chkpt_path
    start_epoch = 0
    acc_in_ep = []
    pert_count = args.pert_count
    epochs = args.epochs
    dataset = DatasetEnum(args.dataset)
    attack_type = AttackType(args.attack)
    loss = F.cross_entropy
    epsilon, step_size = get_attack_params(dataset)
    epsilon = args.epsilon if not None else epsilon
    step_size = args.step_size if not None else step_size
    lr = get_dataset_default_lr(dataset) if args.lr == -1 else args.lr
    use_cuda = torch.cuda.is_available()
    model, optimizer = utils.init(dataset, lr, use_cuda)
    train_perturbation = FGSMAttack(loss, model, epsilon).perturb
    os.environ['seed'] = str(args.seed)

    if args.model_path != "":
        test_loader = utils.only_test(dataset, args.batch_size)
        model, optimizer, start_epoch = utils.load_check_point("./checkpoint/" + args.model_path, model,
                                                               use_cuda, optimizer)
        test(test_loader, model, use_cuda, dataset)
    else:
        pert_str = f'{args.pert_file} file' if args.pert_file is not None else f'random sample with seed {args.seed}'
        print(f"Training saved in {checkpoint_path}. Using {args.pert_count} perturabtions in range: {args.pert_range}, via {pert_str} on dataset {dataset}")
        train_loader, test_loader = load_data(dataset, args.pert_file, pert_count, batch_size, args.pert_range)
        train_loss, train_acc, test_loss, test_acc = train_model(train_loader, test_loader, use_cuda, epochs, model,
                                                                 optimizer, dataset, loss)


    if attack_type == AttackType.FGSM_ATTACK:
        acc = fgsm_attack(test_loader, model, use_cuda, loss, test_batch, init_ep=epsilon, max_range=2)
    elif attack_type == AttackType.PGD_ATTACK:
        acc, corr, total = invoke_pgd_attack(test_loader, model, use_cuda, epsilon, step_size, loss, test_batch, iters=args.pgd_iters)
