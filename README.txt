Requirements:
Python 3.6.8
numpy 1.19.4+mkl
torchvision 0.3.0
torch 1.1.0
scikit-learn 0.24.2
scipy 1.5.4
matlabengineforpython R2018a

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