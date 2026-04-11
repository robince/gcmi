% GCMI Gaussian-copula mutual information estimators.
%
% Core helpers:
%   setup_gcmi                 Add matlab/ (and optionally matlab_examples/) to the MATLAB path.
%   gcmi_version               Print the package version.
%
% Continuous estimators:
%   gcmi_cc                    Gaussian-copula MI, continuous/continuous
%   gccmi_ccc                  Gaussian-copula CMI, continuous/continuous/continuous
%   gccmi_ccd                  Gaussian-copula CMI, continuous/continuous/discrete
%   mi_gg                      Gaussian MI
%   cmi_ggg                    Gaussian conditional MI
%   ent_g                      Gaussian entropy
%
% Continuous/discrete estimators:
%   gcmi_model_cd              Gaussian-copula MI, continuous/discrete
%   gcmi_mixture_cd            Gaussian-copula MI, continuous/discrete
%   mi_model_gd                Gaussian MI, continuous/discrete
%   mi_mixture_gd              Gaussian MI, continuous/discrete
%
% Vectorized estimators:
%   mi_gg_vec                  Vectorized Gaussian MI
%   mi_model_gd_vec            Vectorized Gaussian/discrete MI
%   mi_mixture_gd_vec          Vectorized Gaussian/discrete MI
%   cmi_ggg_vec                Vectorized Gaussian conditional MI
%
% Utilities:
%   ctransform                 Empirical copula transform
%   bias_cmi_ggg_bits          Analytic Gaussian conditional MI bias term in bits
%   bias_ent_g_bits            Analytic Gaussian entropy bias term in bits
%   bias_mi_gd_bits            Analytic Gaussian/discrete MI bias term in bits
%   bias_mi_gg_bits            Analytic Gaussian/Gaussian MI bias term in bits
%   copnorm                    Copula normalization
%   maxstar                    Stable log-sum-exp helper
%   vecchol                    Vectorized Cholesky decomposition
%   validate_discrete_labels   Discrete label validation helper
