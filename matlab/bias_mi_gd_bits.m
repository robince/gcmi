function bias_bits = bias_mi_gd_bits(Xdim, Ntrl, bincount)
%BIAS_MI_GD_BITS Analytic bias correction for Gaussian/discrete MI.
%   bias_bits = bias_mi_gd_bits(Xdim, Ntrl, bincount) returns the
%   scalar bias correction in bits for the model-based MI estimator between an
%   Xdim-dimensional Gaussian variable and a discrete variable with class
%   counts specified by bincount.

if ~isscalar(Xdim) || ~isfinite(Xdim) || Xdim < 1 || Xdim ~= floor(Xdim)
    error('bias_mi_gd_bits:Xdim', 'Xdim must be a positive integer scalar');
end
if ~isscalar(Ntrl) || ~isfinite(Ntrl) || Ntrl < 1 || Ntrl ~= floor(Ntrl)
    error('bias_mi_gd_bits:Ntrl', 'Ntrl must be a positive integer scalar');
end
if ~isvector(bincount) || isempty(bincount)
    error('bias_mi_gd_bits:bincount', 'bincount must be a non-empty vector');
end

Ntrl_y = double(bincount(:)');
if any(~isfinite(Ntrl_y)) || any(Ntrl_y < 1) || any(Ntrl_y ~= floor(Ntrl_y))
    error('bias_mi_gd_bits:bincount', ...
        'bincount must contain positive integer class counts');
end
if sum(Ntrl_y) ~= double(Ntrl)
    error('bias_mi_gd_bits:bincount', ...
        'sum(bincount) must equal Ntrl');
end
if Ntrl <= Xdim
    error('bias_mi_gd_bits:samples', ...
        'Ntrl must be greater than Xdim');
end
if any(Ntrl_y <= Xdim)
    error('bias_mi_gd_bits:samples', ...
        'Each class count must be greater than Xdim');
end

w = Ntrl_y / Ntrl;
bias_unc = bias_ent_g_bits(Xdim, Ntrl);
bias_cond = zeros(1, numel(Ntrl_y));
for yi = 1:numel(Ntrl_y)
    bias_cond(yi) = bias_ent_g_bits(Xdim, Ntrl_y(yi));
end
bias_bits = bias_unc - sum(w .* bias_cond);
