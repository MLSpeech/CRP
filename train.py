from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def train(loader, model, optimizer, epoch, cuda, verbose=True, verbose_interval=1000):
    model.train()
    total = len(loader.sampler)
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % verbose_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{total} ({round(100. * batch_idx / len(loader))}%)] "
                  f"Loss: {round(loss.item(),6)}")

    train_loss /= total
    print(f"\nTrain Set: Average loss: {round(train_loss,4)} Accuracy {correct}/{total} ({round(100. * correct.item() / total,4)}%)\n")

    return train_loss, 100. * correct.item() / total


def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_func = F.cross_entropy
    for data, target in loader:
        total, test_loss, correct, corr_items_index = test_batch(data, target, total, cuda, model, test_loss, correct,
                                                                 loss_func)
    test_loss /= len(loader.dataset)

    if verbose:
        print(f"\nTest Set: Average loss: {round(test_loss,6)} Accuracy {correct}/{total} ({round(100. * correct / total,4)})\n")

    return test_loss, float(correct) / total


def test_batch(data, target, total, cuda, model, test_loss, correct, loss_func):
    model.eval()
    total += data.shape[0]

    with torch.no_grad():
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += loss_func(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        corr_items_index = (pred.eq(target.data.view_as(pred)).squeeze() == 1).nonzero()
    return total, test_loss, correct.item(), corr_items_index
