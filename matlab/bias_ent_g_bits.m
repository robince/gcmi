function bias_bits = bias_ent_g_bits(Nvar, Ntrl)
%BIAS_ENT_G_BITS Analytic Gaussian entropy bias in bits.
%   bias_bits = bias_ent_g_bits(Nvar, Ntrl) returns the
%   finite-sample bias term for the log-determinant part of the entropy of an
%   Nvar-dimensional Gaussian estimated from Ntrl samples.

if ~isscalar(Nvar) || ~isfinite(Nvar) || Nvar < 1 || Nvar ~= floor(Nvar)
    error('bias_ent_g_bits:Nvar', ...
        'Nvar must be a positive integer scalar');
end
if ~isscalar(Ntrl) || ~isfinite(Ntrl) || Ntrl < 1 || Ntrl ~= floor(Ntrl)
    error('bias_ent_g_bits:Ntrl', ...
        'Ntrl must be a positive integer scalar');
end
if Ntrl <= Nvar
    error('bias_ent_g_bits:samples', ...
        'Ntrl must be greater than Nvar');
end

ln2 = log(2);
dterm = (ln2 - log(Ntrl - 1)) / 2;
psiterms = sum(psi((Ntrl - (1:Nvar)) / 2) / 2);
bias_bits = (Nvar * dterm + psiterms) / ln2;
