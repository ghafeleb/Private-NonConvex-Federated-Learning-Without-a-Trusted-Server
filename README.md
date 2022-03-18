![Non-Convex Local Differential Privacy Diagram](figures/NC_LDP_diagram_v2.png "Non-Convex Local Differential Privacy Diagram" = 250x250)

This repository contains the implementations for the following paper:\
Andrew Lowy, Ali Ghafelebashi, Meisam Razaviyayn, [Private Non-Convex Federated Learning Without a Trusted Server(https://arxiv.org/abs/2110.11205.pdf), arXiv 2022.

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

As noises are generated using MATLAB, installed MATLAB on the system is required. noise_generator_localSGD2.m is used to generate noises for noisy local SGD and no modification is required for it. main_FL.m calls the function from noise_generator.m to generate the noise for noisy SPIDER and noisy minibatch SGD. List of epsilons, list of clipping thresholds (Ls), n, list of R values (Rs), and M should be modifies based on the experiment setting.

We have run the code with the following four settings:
R=25, Mavail=25
R=25, Mavail=12
R=50, Mavail=25
R=50, Mavail=12

The other parameters were set in all experiments as follows:
np_seed = 1
q = 1/7
seedModel = 1
seedNoise = 1
epsilons = [0.75, 1, 1.5, 3, 6, 12, 18]
Ls = [1, 5, 10, 100, 10000]
K_constant = 24
n_stepsizes = 5

Run the code with the following command:
python DP_FL.py

## Citation
If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{??,
  title={Private Non-Convex Federated Learning Without a Trusted Server},
  author={??},
  booktitle={??},
  year={2022}
}
```
