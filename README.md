<a href="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/NC_LDP_diagram_v2.png"><img src="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/NC_LDP_diagram_v2.png" align="center" width="400" ></a>

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
If you want to generate noises using MATLAB, installed MATLAB on the system is required. 

## Model Training:
To train the model with generated noises by MATLAB, run the code with the following command:
```bash
python DP_FL.py 
```

To train the model with generated noises by Python, run the code with the following command:
```bash
python DP_FL2.py 
```

