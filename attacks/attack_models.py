import pickle
import torch.nn.functional as F
from train import test_batch
from utils import load_check_point, init
from .attacks import fgsm_attack,invoke_pgd_attack


def white_box_eval(files, dataset, test_loader, epsilon, alpha):
    '''

    :param files: list of models path to evaluate
    :return:
    '''

    res=[]

    model, optimizer = init(dataset, 0.001)
    for f in files:
        print(f"attacking {f}")
        model, optimizer,_ = load_check_point(f, model,
                                            optimizer)
        accs=[]
        for it in [20,40,100]:
            accs.append(invoke_pgd_attack(test_loader,model,True,epsilon,alpha,F.cross_entropy,test_batch,iters=it))
        res.append(accs)

    print(f"res:\n {res}")

def black_box_eval(files, dataset, test_loader, epsilon, alpha):
    '''
    list of models to use for black-box attacks
    :param duos:
    :return:
    '''
    res=[]

    model_s, optimizer_s, use_cuda_s = init(dataset, 0.001)
    model_t, optimizer_t, use_cuda_t = init(dataset, 0.001)
    for i,target in enumerate(files):
        sres = [target,[],[],[]]
        for j, source in enumerate(files):
            if j == i :
                continue
            print(f"attacking {target} from {source}")
            model_s, optimizer_s,_ = load_check_point(source, model_s,
                                                optimizer_s)
            model_t, optimizer_t,_ = load_check_point(target, model_t,
                                                optimizer_t)

            accs = [target]
            try:
                print("running fgsm")
                accs.append(fgsm_attack(test_loader, model_s, True, F.cross_entropy,test_batch,init_ep=epsilon,max_range=1,target_model=model_t))
                for it in [20,100]:
                    accs.append(invoke_pgd_attack(test_loader,model_s,True,epsilon,alpha,F.cross_entropy,test_batch,iters=it,target_model=model_t))
                res.append(accs)
            except Exception as e:
                print(f"Exception model: {e}")
                res.append(sres)
                continue
    print(f"res:\n {res}")
    with open(f"blackboxdataset{dataset}", "wb") as f:
        pickle.dump(res, f)
