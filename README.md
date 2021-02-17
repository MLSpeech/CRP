
# Models will be published.
# Training the models

Optional Arguments:
--attack is the attack the last model will be attacked with
--pgd_iters if attack is 1 (PGD) number of PGD iterations
--epsilon attack epsilon
--step_size attack step size
  Perturbations:
  If a .npy file will be specified in --pert_file it will train with it
  If not, a seed (can also be specified with --seed or a random one is chosen) is used and they are sampled from a uniform distribution.
# Training MNIST
Run the following to train the model:
```setup
python crp_train.py  --dataset 0 --batch_size 128 --pert_count 10 --pert_range 0.3 --epochs 50 --chkpt_path mnist_model.pth 
```

# Training SVHN
Run the following to train the model:
```setup
python crp_train.py --dataset 2 --batch_size 128 --pert_count 10 --pert_range 0.0313 --epochs 350 --chkpt_path svhn_model.pth --seed 323
```

# Training CIFAR-10
Run the following to train the model:
```setup
python crp_train.py  --dataset 1 --batch_size 128 --pert_count 10 --pert_range 0.06274 --epochs 700 --chkpt_path cifar10_model.pth
```

# Evaluation
Instead of --chkpt_path use the argument --model_path. Use --attack 0 for FGSM and --attack 1 for PGD.
For example, for CIFAR-10:
```setup
python crp_train.py --dataset 1 --batch_size 128 --pert_count 10 --pert_range 0.06274 --epochs 700 --model_path cifar10_model.pth --attack 1
```

# Results
Evaluation of our best model against PGD (100 iterations) and FGSM
| Dataset    |      FGSM       |      PGD       |
| -----------|---------------- | -------------- |
| MNIST      |     98.89%      |      99.49%    |
| SVHN 20pert|     81.84%      |      80.21%    |
| CIFAR-10   |     66.75%      |      51.63%    |
