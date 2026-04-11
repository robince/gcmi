function I = mi_model_gd(x, y, Ym, biascorrect, demeaned)
% MI_MODEL_GD Mutual information (MI) between a Gaussian and a discrete
%         variable in bits bits based on ANOVA style model comparison.
%   I = mi_model_gd(x,y,Ym) returns the MI between the (possibly multidimensional)
%   Gaussian variable x and the discrete variable y.
%   For 1D x this is a lower bound to the mutual information.
%   Rows of x correspond to samples, columns to dimensions/variables. 
%   (Samples first axis)
%   y should contain integer values in the range [0 Ym-1] (inclusive).
%
%   biascorrect : true / false option (default true) which specifies whether
%   bias correction should be applied to the esimtated MI.
%   demeaned : false / true option (default false) which specifies whether the
%   input data already has zero mean (true if it has been copula-normalized)
%   See also: MI_MIXTURE_GD

% ensure samples first axis for vectors
if isvector(x)
    x = x(:);
end
if ndims(x)~=2
    error('mi_model_gd: input arrays should be 2d')
end
if isvector(y)
    y = y(:);
else
    error('mi_model_gd: only univariate discrete variable supported');
end

Ntrl = size(x,1);
Nvar = size(x,2);

if size(y,1) ~= Ntrl
    error('mi_model_gd: number of trials do not match');
end

validate_discrete_labels(y, Ym, 'mi_model_gd', 'y');

% default option values
if nargin<4
    biascorrect = true;
end
if nargin<5
    demeaned = false;
end

if ~demeaned
    x = bsxfun(@minus,x,sum(x,1)/Ntrl);
end

% class-conditional entropies
Ntrl_y = zeros(1,Ym);
Hcond = zeros(1,Ym);
for yi=1:Ym
    idx = y==(yi-1);
    xm = x(idx,:);
    Ntrl_y(yi) = size(xm,1);
    xm = bsxfun(@minus,xm,sum(xm,1)/Ntrl_y(yi));
    Cm = (xm'*xm) / (Ntrl_y(yi) - 1);
    chCm = chol(Cm);
    Hcond(yi) = sum(log(diag(chCm)));% + c*Nvar;
end
% class weights
w = Ntrl_y ./ Ntrl;

% unconditional entropy from unconditional Gaussian fit
Cx = (x'*x) / (Ntrl-1);
chC = chol(Cx);
Hunc = sum(log(diag(chC)));% + c*Nvar;


% apply bias corrections
ln2 = log(2);
if biascorrect
    I = Hunc - sum(w .* Hcond);
    I = (I / ln2) - bias_mi_gd_bits(Nvar, Ntrl, Ntrl_y);
    return
end

I = Hunc - sum(w .* Hcond);
% convert to bits
I = I / ln2;
