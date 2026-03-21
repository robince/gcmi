function y = maxstar(x, w)
% MAXSTAR Stable log-sum-exp helper.
%   For vectors, maxstar(x) is equivalent to log(sum(exp(x))).
%   For matrices and N-D arrays, maxstar operates along the first
%   non-singleton dimension.
%
%   maxstar(x,w) is the log of a weighted sum of exponentials,
%   equivalent to log(sum(w.*exp(x))). In this codebase w is expected
%   to be a vector with one weight per row of x.

if nargin<2
    w = [];
else
    if ~isvector(w)
        error('maxstar: w must be a vector')
    end
    if length(w) ~= size(x,1)
        error('maxstar: weight does not match x')
    end
end
%%
w = w(:);
szx = size(x);
if isempty(w)
    % no weight
    m = max(x);
    y = m + log(sum(exp(bsxfun(@minus,x,m))));
else
    % Move the weight into the exponent xw and find
    % m = max(xw) over terms with positive weights
    wpos = w>0;
    xw = bsxfun(@plus, x(wpos,:), log(w(wpos)));
    m = max(xw);
    exwp = exp( bsxfun(@minus, xw, m) );
    wneg = w<0;
    exwn = exp( x(wneg,:) + bsxfun(@minus,log(-w(wneg)), m) );
    y = m + log(sum(exwp,1) - sum(exwn,1));
end
y = reshape(y,[szx(2:end) 1]);



