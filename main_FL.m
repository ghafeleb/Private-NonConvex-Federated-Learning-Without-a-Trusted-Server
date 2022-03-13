clear all
close all
clc
%% Initialization
epsilons = [0.75, 1, 1.5, 3, 6, 12, 18];
Ls = [1, 5, 10, 100, 10000]; 
max_eps = max(epsilons);
layerSizes = [3200, 64, 128, 2];
layerNames = ["w1", "b1", "w2", "b2"];
layerStr = strcat(layerNames(1, 1), "_", num2str(layerSizes(1, 1)));
for layerIdx=2:size(layerSizes, 2)
    layerStr = strcat(layerStr, "_", layerNames(1, layerIdx), "_", num2str(layerSizes(1, layerIdx)));
end
n = 1238;
delta = 1/(n^2);
Rs = [25, 50];
% M = 25;
M = 12;
outputFolderHead = fullfile("data");
mkdir(outputFolderHead);
outputFolderHead = fullfile("data", "noise");
mkdir(outputFolderHead);
for rIdx=1:size(Rs, 2)
    R = Rs(rIdx);
    for seedNoise=1:1
        %% Generate Noise for noisy SPIDER and noisy MB SGD
        % Folder
        outputFolderHead = fullfile("data", "noise", layerStr);
        mkdir(outputFolderHead);
        folderTemp = strcat("delta", num2str(delta, 3), "_", ...
                            "n", num2str(n), "_", ...
                            "R", num2str(R), "_", ...
                            "M", num2str(M), "_", ...
                            "sN", num2str(seedNoise));
        folderPath1 = fullfile(outputFolderHead, folderTemp);
        mkdir(folderPath1);

        % Noises
        for idxL=1:size(Ls, 2)
            for idxEps=1:size(epsilons, 2)
                eps = epsilons(1, idxEps);
                L = Ls(1, idxL);
                folderPath2 = fullfile(folderPath1, strcat("eps_", num2str(eps), "_L_", num2str(L)))
                mkdir(folderPath2);
                for layerIdx=1:size(layerSizes, 2)
                    layerName = layerNames(1, layerIdx);
                    layerSize = layerSizes(1, layerIdx);
                    noise_generator(layerName, layerSize, eps, delta, n, R, L, M, folderPath2, seedNoise);
                end
            end
        end
    end
end