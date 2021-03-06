<a href="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/NC_LDP_diagram_v2.png"><img src="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/NC_LDP_diagram_v2.png" align="center" width="400" ></a>

This repository contains the implementations for the following paper:\
Andrew Lowy, Ali Ghafelebashi, Meisam Razaviyayn, [Private Non-Convex Federated Learning Without a Trusted Server(https://arxiv.org/abs/2203.06735), arXiv 2022.

We study differentially private (DP) federated learning (FL) with non-convex loss functions and heterogeneous (non-i.i.d.) client data in the absence of a trusted server, both with and without a secure "shuffler" to anonymize client reports. We propose new algorithms that satisfy local differential privacy (LDP) at the client level and shuffle differential privacy (SDP) for three classes of Lipschitz loss functions. First, we consider losses satisfying the Proximal Polyak-Łojasiewicz (PL) inequality, which is an extension of the classical PL condition to the constrained setting. Prior works studying DP Lipschitz, PL optimization only consider the unconstrained problem, which rules out many interesting practical losses (e.g. strongly convex, least squares, regularized logistic regression). We propose LDP and SDP algorithms that nearly attain the optimal strongly convex, homogeneous (i.i.d.) rates. Second, we provide the first DP algorithms for non-convex/non-smooth loss functions. Third, we specialize to smooth, unconstrained non-convex FL. Our bounds improve on the state-of-the-art, even in the special case of a single client, and match the non-private lower bound in practical parameter regimes. Numerical experiments show that our algorithm yields better accuracy than baselines for most privacy levels.

The code and experiments to reproduce our numerical results are provided in this repository.

## Requirements:
- Python 3.6.8
- numpy 1.19.4+mkl
- torchvision 0.3.0
- torch 1.1.0
- scikit-learn 0.24.2
- scipy 1.5.4
- matlabengineforpython R2018a

Use the follwoing command to install the dependencies:
```bash
pip install -r requirements.txt
```
As noises are generated using MATLAB, installed MATLAB on the system is required. 

## Model Training:
Run the code with the following command:
```bash
python DP_FL.py --model=MLP --gpu=-1 --clipping=1 --num_classes=2 --dH=64
```

## Citation
If you find this repository useful in your research, please cite the following paper:
```
@article{lowy2022private,
  title={Private Non-Convex Federated Learning Without a Trusted Server},
  author={Lowy, Andrew and Ghafelebashi, Ali and Razaviyayn, Meisam},
  journal={arXiv preprint arXiv:2203.06735},
  year={2022}
}
```
