% This script provides a synthetic max-statistics example for the optimized
% C++ MEX implementation of continuous/discrete mutual information.
%
% The direct native entrypoint is:
%   I = info_cd_slice_cpp(X, Xdim, Y, Ym, Ntrl, Nthread)
%
% This script uses cell mode - cells are delimited by %% lines and can
% be run with:
% ctrl-enter (windows, linux) or cmd-enter (mac)
% or "Run Section" button from the toolbar
% or right click -> Evaluate current section
%
% Add the repository root to your MATLAB path and run:
%   setup_gcmi('IncludeExamples', true)
%
% Questions / comments : robince@gmail.com

%% Check that the native runtime is available

if exist('info_cd_slice_cpp', 'file') ~= 3
    error(['This example requires the compiled native C++ MEX runtime. ' ...
           'Run setup_gcmi from a MATLAB release with matching binaries ' ...
           'under matlab/cpp_mex/bin, or compile matlab/cpp_mex first.']);
end

runtime = gcmi_cpp_ping()

%% Build a 1000-page synthetic problem with sparse class effects

rng(1);

Ym = 3;
NtrlPerClass = 60;
Ntrl = Ym * NtrlPerClass;
Npage = 1000;
Xdim = 2;
Nperm = 100;
Nthread = 2;

% The direct C++ kernel expects labels 0 .. Ym-1.
labels = repelem(int32(0:Ym-1), NtrlPerClass)';
labels = labels(randperm(Ntrl));

effectStrength = zeros(Npage, 1);
centers = [180 470 725 915];
amps = [0.50 0.75 0.65 0.80];
width = 10;
for ci = 1:numel(centers)
    idx = max(1, centers(ci)-width):min(Npage, centers(ci)+width);
    shape = amps(ci) * exp(-((idx - centers(ci)).^2) / (2 * (width/2)^2));
    effectStrength(idx) = max(effectStrength(idx), shape(:));
end
effectPages = effectStrength > 0.10;

% Generate Gaussian features with class-specific mean offsets on sparse pages.
noiseCov = [1.00 0.30; 0.30 1.00];
noiseChol = chol(noiseCov, 'lower');
classOffsetTemplate = [-1.10  0.35; ...
                        0.00  0.00; ...
                        0.95 -0.30];
classOffsets = classOffsetTemplate(double(labels) + 1, :);

x = zeros(Ntrl, Npage, Xdim);
for page = 1:Npage
    noise = randn(Ntrl, Xdim) * noiseChol;
    x(:, page, :) = noise + effectStrength(page) * classOffsets;
end

% The direct native kernel does not remove the overall mean internally.
x = bsxfun(@minus, x, mean(x, 1));

%% Run the observed native analysis and a MATLAB reference check

% MATLAB vectorized reference layout: [Ntrl, Npage, Xdim]
Iref = mi_model_gd_vec(x, labels, Ym, false, true);

% Native layout for info_cd_slice_cpp: [Xdim, Ntrl, Npage]
nativeX = permute(x, [3 1 2]);
tic
Iobs = info_cd_slice_cpp(nativeX, Xdim, labels, Ym, Ntrl, Nthread);
obsTimeSeconds = toc;

fprintf('Observed scan: %d pages, %d-D X, %.3f s\n', Npage, Xdim, obsTimeSeconds);
fprintf('Observed native/reference max abs diff: %.3g bits\n', max(abs(Iobs(:) - Iref(:))));

%% Max-statistics with 100 label permutations

permIndex = zeros(Ntrl, Nperm);
for pi = 1:Nperm
    permIndex(:, pi) = randperm(Ntrl);
end

IpermMax = zeros(Nperm, 1);
tic
for pi = 1:Nperm
    labelPerm = labels(permIndex(:, pi));
    Iperm = info_cd_slice_cpp(nativeX, Xdim, labelPerm, Ym, Ntrl, Nthread);
    IpermMax(pi) = max(Iperm);
end
permTimeSeconds = toc;

threshold = prctile(IpermMax, 95);
isSignificant = Iobs > threshold;
truePositivePages = sum(isSignificant(:) & effectPages(:));
falsePositivePages = sum(isSignificant(:) & ~effectPages(:));
missedEffectPages = sum(~isSignificant(:) & effectPages(:));

summary = table( ...
    Npage, ...
    sum(effectPages), ...
    sum(isSignificant), ...
    truePositivePages, ...
    falsePositivePages, ...
    missedEffectPages, ...
    threshold, ...
    permTimeSeconds, ...
    'VariableNames', {'Npage', 'EffectPages', 'SignificantPages', ...
                      'TruePositives', 'FalsePositives', 'MissedEffects', ...
                      'MaxStatThreshold_bits', 'PermutationTime_seconds'});
disp(summary)

[~, topPages] = maxk(Iobs(:), 10);
topPages = reshape(topPages, [], 1);
topEffect = reshape(effectStrength(topPages), [], 1);
topObserved = reshape(Iobs(topPages), [], 1);
topSignificant = reshape(isSignificant(topPages), [], 1);
topHits = table(topPages, topEffect, topObserved, topSignificant, ...
    'VariableNames', {'Page', 'EffectStrength', 'ObservedMI_bits', 'Significant'});
disp(topHits)

%% Plot the observed MI and the max-stat threshold

figure
plot(1:Npage, Iobs, 'b-', 'LineWidth', 1)
hold on
yline(threshold, 'r--', 'LineWidth', 1.5)
plot(find(effectPages), Iobs(effectPages), 'ko', 'MarkerSize', 4)
plot(find(isSignificant), Iobs(isSignificant), 'gx', 'MarkerSize', 5, 'LineWidth', 1)
xlabel('Page index')
ylabel('Mutual information (bits)')
title('Discrete native max-stat example with sparse class effects')
legend('Observed MI', '95% max-stat threshold', 'Planted effect pages', ...
       'Significant pages', 'Location', 'northwest')
grid on

%% Notes

% For direct calls to info_cd_slice_cpp:
%   - X must be [Xdim, Ntrl, Npage]
%   - labels must be zero-based: 0 .. Ym-1
%   - empty classes are rejected
%   - the direct kernel expects approximately Gaussian continuous inputs
%   - this example generates Gaussian class-conditional data directly
%   - mean-center X before the direct native call
%   - for non-Gaussian inputs, apply copnorm before calling the native kernel
