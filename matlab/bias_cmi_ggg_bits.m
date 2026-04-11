function bias_bits = bias_cmi_ggg_bits(Nvarx, Nvary, Nvarz, Ntrl)
%BIAS_CMI_GGG_BITS Analytic bias correction for Gaussian conditional MI.
%   bias_bits = bias_cmi_ggg_bits(Nvarx, Nvary, Nvarz, Ntrl) returns
%   the scalar bias correction in bits for CMI(X;Y|Z) where X, Y, and Z are
%   Gaussian variables with dimensions Nvarx, Nvary, and Nvarz.

if ~isscalar(Nvarx) || ~isfinite(Nvarx) || Nvarx < 1 || Nvarx ~= floor(Nvarx)
    error('bias_cmi_ggg_bits:Nvarx', ...
        'Nvarx must be a positive integer scalar');
end
if ~isscalar(Nvary) || ~isfinite(Nvary) || Nvary < 1 || Nvary ~= floor(Nvary)
    error('bias_cmi_ggg_bits:Nvary', ...
        'Nvary must be a positive integer scalar');
end
if ~isscalar(Nvarz) || ~isfinite(Nvarz) || Nvarz < 1 || Nvarz ~= floor(Nvarz)
    error('bias_cmi_ggg_bits:Nvarz', ...
        'Nvarz must be a positive integer scalar');
end

bias_bits = bias_ent_g_bits(Nvarx + Nvarz, Ntrl) ...
    + bias_ent_g_bits(Nvary + Nvarz, Ntrl) ...
    - bias_ent_g_bits(Nvarx + Nvary + Nvarz, Ntrl) ...
    - bias_ent_g_bits(Nvarz, Ntrl);
