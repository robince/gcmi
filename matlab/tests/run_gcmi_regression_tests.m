function results = run_gcmi_regression_tests()
%RUN_GCMI_REGRESSION_TESTS Deterministic smoke tests for the MATLAB package.
%
%   results = run_gcmi_regression_tests() executes a small set of regression
%   checks covering the public wrappers, helper fixes, and discrete-class
%   validation behavior. The function prints a short summary and throws an
%   error if any test fails.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(repoRoot);
setup_gcmi();

tests = {
    @test_gcmi_cc_wrapper
    @test_gccmi_ccc_wrapper
    @test_gccmi_ccd_wrapper
    @test_vecchol_4x4
    @test_maxstar
    @test_empty_class_validation
};

results = struct('passed', 0, 'failed', 0, 'details', {{}}); 
for k = 1:numel(tests)
    testName = func2str(tests{k});
    try
        feval(tests{k});
        results.passed = results.passed + 1;
        results.details{end+1} = sprintf('PASS %s', testName);
    catch ME
        results.failed = results.failed + 1;
        results.details{end+1} = sprintf('FAIL %s: %s', testName, ME.message);
    end
end

for k = 1:numel(results.details)
    disp(results.details{k});
end
fprintf('GCMI regression summary: %d passed, %d failed\n', results.passed, results.failed);

if results.failed > 0
    error('run_gcmi_regression_tests: some regression checks failed')
end

end

function test_gcmi_cc_wrapper()
x = [0.1; 1.4; 0.7; 2.5; 3.1];
y = [-1.2; 0.4; 2.0; 1.1; -0.2];
expected = mi_gg(copnorm(x), copnorm(y), true, true);
actual = gcmi_cc(x, y);
assert_close(actual, expected, 1e-12, 'gcmi_cc wrapper');
end

function test_gccmi_ccc_wrapper()
x = [0.2; 1.3; 2.1; 3.7; 4.4];
y = [-0.7; 0.9; 1.8; 2.6; 3.5];
z = [1.0; 2.2; 3.3; 4.1; 5.4];
expected = cmi_ggg(copnorm(x), copnorm(y), copnorm(z), true, true);
actual = gccmi_ccc(x, y, z);
assert_close(actual, expected, 1e-12, 'gccmi_ccc wrapper');
end

function test_gccmi_ccd_wrapper()
x = [0.5; 1.4; 2.3; 3.2; 0.8; 1.7; 2.9; 3.6];
y = [3.1; 2.4; 3.3; 1.2; 1.5; 2.8; 1.9; 3.4];
z = [0; 0; 0; 0; 1; 1; 1; 1];
[cmi, jointI] = gccmi_ccd(x, y, z, 2);

Icond = zeros(2,1);
cx = cell(2,1);
cy = cell(2,1);
for zi = 1:2
    idx = z == (zi-1);
    thsx = copnorm(x(idx,:));
    thsy = copnorm(y(idx,:));
    Icond(zi) = mi_gg(thsx, thsy, true, true);
    cx{zi} = thsx;
    cy{zi} = thsy;
end
expectedCmi = ([sum(z==0); sum(z==1)] / numel(z)).' * Icond;
expectedJoint = mi_gg(cell2mat(cx), cell2mat(cy), true, false);
assert_close(cmi, expectedCmi, 1e-12, 'gccmi_ccd conditional MI');
assert_close(jointI, expectedJoint, 1e-12, 'gccmi_ccd pooled MI');
end

function test_vecchol_4x4()
base = [4 1 0 0; 1 3 1 0; 0 1 2 1; 0 0 1 2];
cov = base' * base;
expected = chol(cov);
actual = squeeze(vecchol(reshape(cov, [1 4 4])));
assert_close(actual, expected, 1e-12, 'vecchol 4x4 branch');
end

function test_maxstar()
x = [0; -1; -2];
w = [1; 2; 3];
assert_close(maxstar(x), log(sum(exp(x))), 1e-12, 'maxstar unweighted');
assert_close(maxstar(x, w), log(sum(w .* exp(x))), 1e-12, 'maxstar weighted');
end

function test_empty_class_validation()
x = [0; 1; 2; 3];
y = [0; 0; 1; 1];
z = [0; 0; 1; 1];
expect_error(@() mi_model_gd(x, y, 3), 'empty discrete classes are not supported');
expect_error(@() mi_mixture_gd(x, y, 3), 'empty discrete classes are not supported');
expect_error(@() gcmi_model_cd(x, y, 3), 'empty discrete classes are not supported');
expect_error(@() gcmi_mixture_cd(x, y, 3), 'empty discrete classes are not supported');
expect_error(@() mi_model_gd_vec(reshape(x, [4 1 1]), y, 3), 'empty discrete classes are not supported');
expect_error(@() mi_mixture_gd_vec(reshape(x, [4 1 1]), y, 3), 'empty discrete classes are not supported');
expect_error(@() gccmi_ccd(x, y, z, 3), 'empty discrete classes are not supported');
end

function assert_close(actual, expected, tolerance, label)
if nargin < 4
    label = 'value';
end
if any(abs(actual(:) - expected(:)) > tolerance)
    error('%s mismatch: expected %s, got %s', label, mat2str(expected), mat2str(actual))
end
end

function expect_error(fun, messageFragment)
didError = false;
try
    fun();
catch ME
    didError = true;
    if isempty(strfind(ME.message, messageFragment))
        error('Unexpected error message: %s', ME.message)
    end
end
if ~didError
    error('Expected error containing "%s" was not raised', messageFragment)
end
end
