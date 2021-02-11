import pickle
import torch
import argparse
import os
from configs import DatasetEnum,get_attack_params
from attacks.attack_models import white_box_eval, black_box_eval
from utils import only_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating models ')
    parser.add_argument('--batch_size', default=300, type=int, help='test loader batch size')
    parser.add_argument('--models_path', type=str, help='path to folder with models')
    parser.add_argument('--dataset', type=int, default=1,
                    help='which dataset to use, 0 for MNIST, 1 for CIFAR10')
    parser.add_argument('--no_white_box', action='store_false')

    args = parser.parse_args()
    dataset = DatasetEnum(args.dataset)
    epsilon, alpha = get_attack_params(dataset)
    args.white_box = bool(args.white_box)
    batch_size = args.batch_size
    files = [f"{args.models_path}/{f}" for f in os.listdir(args.models_path)]

    test_loader, py_dataset = only_test(dataset, batch_size)


    with open("subset_indeces", "rb") as f:
        test_subset_indexes =  pickle.load(f)
    manual_batch_x = None
    manual_batch_y = []

    for i,idx in enumerate(test_subset_indexes[:300]):
        (data, target) = py_dataset.__getitem__(idx)
        manual_batch_x = data.unsqueeze(0) if manual_batch_x is None else torch.cat((manual_batch_x,data.unsqueeze(0)),dim=0)
        manual_batch_y.append(target)
    manual_batch_y = torch.LongTensor(manual_batch_y)
    manual_loader = [[manual_batch_x,manual_batch_y]]

    if args.no_white_box:
        print("whitebox mode")
        white_box_eval(files, dataset, test_loader, epsilon, alpha)
    else:
        print("blackbox mode")
        black_box_eval(files, dataset, test_loader, epsilon, alpha)
