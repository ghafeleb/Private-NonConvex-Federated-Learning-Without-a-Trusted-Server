
<p align="center">
<a href="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/iclr_server_diagram_v2.PNG"><img align="center" src="https://github.com/ghafeleb/Private-NonConvex-Federated-Learning-Without-a-Trusted-Server/blob/master/figures/iclr_server_diagram_v2.PNG" align="center" width="400" ></a>
</p>

We study federated learning (FL)-especially cross-silo FL-with non-convex loss functions  and data from people who do not trust the server or other silos. In this setting, each silo (e.g. hospital) must protect the privacy of each person's data (e.g. patient's medical record), even if the server or other silos act as adversarial eavesdroppers. To that end, we consider inter-silo record-level (ISRL) differential privacy (DP), which requires silo~i's communications to satisfy record/item-level DP. We propose novel ISRL-DP algorithms for FL with heterogeneous (non-i.i.d.) silo data and two classes of Lipschitz continuous loss functions: First, we consider losses satisfying the Proximal Polyak-L ojasiewicz (PL) inequality, which is an extension of the classical PL condition to the constrained setting. In contrast to our result, prior works only considered  unconstrained private  optimization with Lipschitz PL loss, which rules out most interesting PL losses such as strongly convex problems and linear/logistic regression. Our algorithms nearly attain the optimal strongly convex, homogeneous (i.i.d.) rate for ISRL-DP FL without assuming convexity or i.i.d. data. Second, we give the first private algorithms for \textit{non-convex non-smooth} loss functions. Our utility bounds even improve on the state-of-the-art bounds for smooth losses. We complement our upper bounds with lower bounds. Additionally, we provide shuffle DP (SDP) algorithms that improve over the state-of-the-art central DP algorithms 
under more practical trust assumptions. Numerical experiments show that our algorithm has better accuracy than baselines for most rivacy levels.


The code and experiments to reproduce our numerical results are provided in this repository.

## Requirements:
- Python 3.6.8
- numpy 1.19.4+mkl
- torchvision 0.3.0
- torch 1.1.0
- scikit-learn 0.24.2
- scipy 1.5.4
- matlabengineforpython R2018a
- pandas

Use the follwoing command to install the dependencies:
```bash
pip install -r requirements.txt
```
