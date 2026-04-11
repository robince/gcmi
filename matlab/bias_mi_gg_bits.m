function bias_bits = bias_mi_gg_bits(Nvarx, Nvary, Ntrl)
%BIAS_MI_GG_BITS Analytic bias correction for Gaussian/Gaussian MI.
%   bias_bits = bias_mi_gg_bits(Nvarx, Nvary, Ntrl) returns the scalar
%   bias correction in bits for MI between Nvarx- and Nvary-dimensional
%   Gaussian variables estimated from Ntrl samples.

if ~isscalar(Nvary) || ~isfinite(Nvary) || Nvary < 1 || Nvary ~= floor(Nvary)
    error('bias_mi_gg_bits:Nvary', ...
        'Nvary must be a positive integer scalar');
end

bias_bits = bias_ent_g_bits(Nvarx, Ntrl) ...
    + bias_ent_g_bits(Nvary, Ntrl) ...
    - bias_ent_g_bits(Nvarx + Nvary, Ntrl);
