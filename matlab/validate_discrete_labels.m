function counts = validate_discrete_labels(y, Ym, funcName, varName)
%VALIDATE_DISCRETE_LABELS Validate integer class labels and reject empty classes.
%   counts = validate_discrete_labels(y, Ym, funcName, varName)
%   checks that y contains integer labels in the range [0 Ym-1] and
%   that every class has at least one sample.

if nargin < 3 || isempty(funcName)
    funcName = mfilename;
end
if nargin < 4 || isempty(varName)
    varName = 'y';
end

if ~isnumeric(y) || ~isreal(y)
    error('%s: discrete variable %s must be a real numeric vector', funcName, varName)
end
if ~isnumeric(Ym) || ~isreal(Ym) || ~isscalar(Ym) || Ym <= 0 || Ym ~= round(Ym)
    error('%s: %s should be a positive integer', funcName, 'Ym')
end
if ~isvector(y)
    error('%s: only univariate discrete variable supported', funcName)
end

y = y(:);
if isempty(y)
    error('%s: %s must not be empty', funcName, varName)
end
if any(round(y) ~= y) || any(y < 0) || any(y > (Ym-1))
    error('Values of discrete variable %s are not correct', varName)
end

counts = accumarray(double(y) + 1, 1, [Ym 1]);
if any(counts == 0)
    error('%s: empty discrete classes are not supported', funcName)
end
