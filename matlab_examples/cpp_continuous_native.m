% This script provides a synthetic max-statistics example for the optimized
% C++ MEX implementation of continuous/continuous mutual information.
%
% The direct native entrypoint is:
%   I = info_cc_slice_cpp(X, Xdim, Y, Ntrl, Nthread)
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

if exist('info_cc_slice_cpp', 'file') ~= 3
    error(['This example requires the compiled native C++ MEX runtime. ' ...
           'Run setup_gcmi from a MATLAB release with matching binaries ' ...
           'under matlab/cpp_mex/bin, or compile matlab/cpp_mex first.']);
end

runtime = gcmi_cpp_ping()

%% Build a 1000-page synthetic problem with sparse covariance effects

rng(0);

Ntrl = 180;
Npage = 1000;
Xdim = 3;
Nperm = 100;
Nthread = 2;

% Sparse page effects: only a few short page ranges carry signal.
effectStrength = zeros(Npage, 1);
centers = [120 365 610 860];
amps = [0.36 0.52 0.62 0.48];
width = 8;
for ci = 1:numel(centers)
    idx = max(1, centers(ci)-width):min(Npage, centers(ci)+width);
    shape = amps(ci) * exp(-((idx - centers(ci)).^2) / (2 * (width/2)^2));
    effectStrength(idx) = max(effectStrength(idx), shape(:));
end
effectPages = effectStrength > 0.08;

% Generate a shared Gaussian target and page-wise Gaussian responses.
% For page p, Cov(X_p, Y) is controlled by effectStrength(p).
y = randn(Ntrl, 1);
noiseCov = [1.00 0.35 0.20; 0.35 1.00 0.30; 0.20 0.30 1.00];
noiseChol = chol(noiseCov, 'lower');
loadingTemplate = [1.00 0.70 -0.45];

x = zeros(Ntrl, Npage, Xdim);
for page = 1:Npage
    noise = randn(Ntrl, Xdim) * noiseChol;
    coupling = effectStrength(page) * loadingTemplate;
    x(:, page, :) = noise + y * coupling;
end

% The direct native kernel does not remove means internally.
y = y - mean(y, 1);
x = bsxfun(@minus, x, mean(x, 1));

%% Run the observed native analysis and a MATLAB reference check

% MATLAB vectorized reference layout: [Ntrl, Npage, Xdim]
Iref = mi_gg_vec(x, y, false, true);

% Native layout for info_cc_slice_cpp: [Ntrl, Xdim, Npage]
nativeX = permute(x, [1 3 2]);
tic
Iobs = info_cc_slice_cpp(nativeX, Xdim, y, Ntrl, Nthread);
obsTimeSeconds = toc;

fprintf('Observed scan: %d pages, %d-D X, %.3f s\n', Npage, Xdim, obsTimeSeconds);
fprintf('Observed native/reference max abs diff: %.3g bits\n', max(abs(Iobs(:) - Iref(:))));

%% Max-statistics with 100 permutations

permIndex = zeros(Ntrl, Nperm);
for pi = 1:Nperm
    permIndex(:, pi) = randperm(Ntrl);
end

IpermMax = zeros(Nperm, 1);
tic
for pi = 1:Nperm
    yperm = y(permIndex(:, pi), :);
    Iperm = info_cc_slice_cpp(nativeX, Xdim, yperm, Ntrl, Nthread);
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
title('Continuous native max-stat example with sparse covariance effects')
legend('Observed MI', '95% max-stat threshold', 'Planted effect pages', ...
       'Significant pages', 'Location', 'northwest')
grid on

%% Notes

% For direct calls to info_cc_slice_cpp:
%   - X must be [Ntrl, Xdim, Npage]
%   - Y must be [Ntrl, Ydim]
%   - the direct kernel expects approximately Gaussian variables
%   - this example generates Gaussian data from page-specific covariance models
%   - mean-center the arrays before the direct native call
%   - for non-Gaussian inputs, apply copnorm first
