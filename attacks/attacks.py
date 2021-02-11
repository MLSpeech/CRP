import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class FGSMAttack(object):
    def __init__(self, loss, model, epsilon=None):
        self.model = model.eval()
        self.epsilon = epsilon
        self.loss_fn = loss

    def perturb(self, x_orig, y, epsilons=None):
        global max,min
        """
        Given examples (x_orig, y), returns their adversarial
        counterparts with an attack bounded by epsilon.
        """
        curr_epsilon = self.epsilon
        # Providing epsilons in batch
        if epsilons is not None:
            curr_epsilon = epsilons

        x_adv = np.copy(x_orig)

        X_var = to_var(torch.from_numpy(x_adv), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        self.model.zero_grad()
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()
        x_adv += curr_epsilon * grad_sign
        x_adv = np.clip(x_adv, 0, 1)
        return x_adv

    def attack(self, test_loader, test, model, use_cuda, epsilon = None):
        acc, total, test_loss = 0, 0, 0.0

        for data, target in test_loader:
            x = self.perturb(data, target, epsilon)
            x = torch.from_numpy(x)
            total, test_loss, acc, adv_corr_items_index = test(x, target, total, use_cuda, model, test_loss, acc, self.loss_fn)
            test_loss /= len(test_loader.dataset)
        print(test_loss, acc * 100 / total)
        # print(test_loss, adj_acc * 100 / adj_total)
        return acc * 100 / total, acc,total#adj_acc * 100 / adj_total

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, iters=40, step_size=0.01,
                 random_start=True, num_classes = 10):
        """
        Attack parameter initialization. The attack performs iters steps of
        size step_size, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model.eval()
        self.epsilon = epsilon
        self.iters = iters
        self.step_size = step_size
        self.rand = random_start
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def perturb(self, x_orig, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            x = x_orig + torch.FloatTensor(np.random.uniform(-self.epsilon, self.epsilon, x_orig.shape))
            x = torch.clamp(x, 0., 1.)
        else:
            x = torch.Tensor(np.copy(x_orig))

        for i in range(self.iters):
            X_var = to_var(x, requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu()


            x = np.array(x) + self.step_size * np.array(np.sign(grad))

            x = np.clip(x, x_orig - self.epsilon, x_orig + self.epsilon)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return x

def invoke_pgd_attack(test_loader, model, use_cuda, eps, step_size, loss_func, test_func, iters = 40, target_model= None):
    print("running pgd attack")
    test_loss = 0
    correct = 0
    total = 0
    pgd = LinfPGDAttack(model, epsilon = eps, step_size= step_size, iters= iters)

    for data, target in test_loader:
        x_adv = pgd.perturb(data,target)
        if target_model is None:
            target_model = model
        total, test_loss, correct, adv_corr_items_index = test_func(x_adv, target, total, use_cuda, target_model, test_loss, correct, loss_func)
    test_loss /= len(test_loader.dataset)
    print(f'\nPGD Test set: Average loss: {test_loss}, Accuracy: {correct}/{total} ({100. * correct / total}%)\n')
    return correct *100./total, correct, total

def fgsm_attack(test_loader, model, use_cuda, loss, test_func, init_ep=0.1, max_range=20, target_model = None):
    res = []
    increment_val = float("0."+"0"*(len(str(init_ep).split(".")[1])-1)+"1")
    num_dec_digits = len(str(init_ep).split(".")[1])
    ep = round(init_ep, num_dec_digits)
    for _ in range(max_range):
        print(f"ep:{ep}")
        fgsm = FGSMAttack(loss, model, ep)
        if target_model is None:
            target_model = model
        acc, corr, total = fgsm.attack(test_loader, test_func, target_model, use_cuda)
        ep = round(ep+increment_val,num_dec_digits)
        res.append((acc, corr, total))
    return f"{res}, init_ep: {init_ep}, max_ep: {max_range*init_ep}", res
