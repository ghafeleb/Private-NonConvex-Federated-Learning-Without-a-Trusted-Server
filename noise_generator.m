function noise_generator(layerName, layerSize, eps, delta, n, R, L, M, folderPath, seedNoise)
rng(seedNoise);
if mod(R, M)==0
    n_noise = R;
    total_n_noise = R*M;
else
    n_noise = M;
    total_n_noise = R*M;
end
n_bucket = total_n_noise/n_noise;
for idxBucket=1:n_bucket
    % Noise 1
    noise = mvnrnd(zeros(1, layerSize), (32*(L^2)*R*(log(2/delta))/(n^2 * eps^2))*eye(layerSize), n_noise);
    fileName = strcat(layerName, "_", num2str(layerSize), "_bucket", num2str(idxBucket), "_noise1.mat");
    fullPath = fullfile(folderPath, fileName);
    save(fullPath, "noise");
    clearvars noise

    % Noise 2
    noise = mvnrnd(zeros(1, layerSize), (8*(L^2)*R*(log(2/delta))/(n^2 * eps^2))*eye(layerSize), n_noise);
    fileName = strcat(layerName, "_", num2str(layerSize), "_bucket", num2str(idxBucket), "_noise2.mat");
    fullPath = fullfile(folderPath, fileName);
    save(fullPath, "noise");
    clearvars noise
end
