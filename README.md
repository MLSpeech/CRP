
# Models will be published.
# Training MNIST
Create a numpy array for the perturbations:
```setup
np.save('mnist_pert_file.npy',np.random.uniform(-1,1,(10*50000,1,28,28)))
```
Run the following to train the model:
```setup
python adv_train.py --pert_file mnist_pert_file.npy  --dataset 0 --batch_size 128 --pert_count 10 --pert_range 0.3 --epochs 50 --chkpt_path mnist_model.pth --attack 1
```
# Training CIFAR-10
Create a numpy array for the perturbations:
```setup
np.save('cifar10_pert_file.npy',np.random.uniform(-1,1,(10*50000,3,32,32)))
```
Run the following to train the model:
```setup
python adv_train.py --pert_file cifar10_pert_file.npy  --dataset 1 --batch_size 128 --pert_count 10 --pert_range 0.06274 --epochs 700 --chkpt_path cifar10_model.pth --attack 1
```

# Evaluation
Instead of --chkpt_path run the argument --model_path. Use --attack 0 for FGSM and --attack 1 for PGD.
For example, for CIFAR-10:
```setup
python adv_train.py --pert_file cifar10_pert_file.npy  --dataset 1 --batch_size 128 --pert_count 10 --pert_range 0.06274 --epochs 700 --model_path cifar10_model.pth --attack 1
```

# Results
Evaluation of our best model against PGD (100 iterations) and FGSM
| Dataset    |      FGSM       |      PGD       |
| -----------|---------------- | -------------- |
| MNIST      |     98.89%      |      99.49%    |
| CIFAR-10   |     66.75%      |      51.63%    |
