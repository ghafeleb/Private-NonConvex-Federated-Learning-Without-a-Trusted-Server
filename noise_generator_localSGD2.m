function [noise] = noise_generator_localSGD2(layerSize, eps, delta, n, R, L, n_noise, seedNoise)
rng(seedNoise);
% Noise 2
noise = mvnrnd(zeros(1, layerSize), (8*(L^2)*R*(log(2/delta))/(n^2 * eps^2))*eye(layerSize), n_noise);

