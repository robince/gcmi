function results = gcmi_cpp_mex_test(varargin)
p = inputParser;
p.addParameter('Compile', true, @(x) islogical(x) && isscalar(x));
p.addParameter('LegacyPath', fullfile(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))), 'extern', 'gcmi_mex'), @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

cfg = gcmi_cpp_mex_config();
if opts.Compile
    gcmi_cpp_mex_compile();
end

addpath(cfg.OutputDir);
legacyPath = char(opts.LegacyPath);
if exist(legacyPath, 'dir')
    addpath(legacyPath);
end
addpath(fullfile(cfg.RepoRoot, 'matlab'));

rng(42, 'twister');
    results = struct('passed', 0, 'failed', 0, 'details', {{}});
    tests = {@test_ping, @test_probes, @test_copnorm_slice, @test_info_cc_slice, @test_info_cc_single_page, @test_info_cd_slice, @test_info_cd_single_page};
for i = 1:numel(tests)
    name = func2str(tests{i});
    try
        feval(tests{i});
        results.passed = results.passed + 1;
        results.details{end+1} = sprintf('PASS %s', name); %#ok<AGROW>
    catch ME
        results.failed = results.failed + 1;
        results.details{end+1} = sprintf('FAIL %s: %s', name, ME.message); %#ok<AGROW>
    end
end
for i = 1:numel(results.details)
    disp(results.details{i});
end
fprintf('gcmi_cpp_mex test summary: %d passed, %d failed\n', results.passed, results.failed);
if results.failed > 0
    error('gcmi_cpp_mex_test: failures detected')
end

    function test_ping()
        info = gcmi_cpp_ping();
        assert(isstruct(info));
        assert(strcmp(info.release, cfg.Release));
        assert(strcmp(info.arch, cfg.Arch));
    end

    function test_probes()
        a = [4 1; 1 3];
        expected = sum(log(diag(chol(a' * a))));
        actual = gcmi_cpp_blas_probe(a);
        assert(abs(actual - expected) < 1e-12);
        out = gcmi_cpp_omp_probe(9, 3);
        assert(isequal(out, (1:9).'));
        rt = gcmi_cpp_runtime_probe(randn(8, 5), 2);
        assert(numel(rt) == 5);
    end

    function test_copnorm_slice()
        x = [3.0 1.0 2.0 4.0; 1.5 -1.0 0.0 2.0; 1.1 1.2 2.3 2.4];
        actual = copnorm_slice_cpp(x, 2);
        if exist('copnorm_slice_omp_c_double', 'file') == 3
            legacy = copnorm_slice_omp_c_double(x, 2);
            assert(max(abs(actual(:) - legacy(:))) < 1e-12);
        else
            expected = zeros(size(x));
            for page = 1:size(x, 2)
                expected(:, page) = copnorm(x(:, page));
            end
            assert(max(abs(actual(:) - expected(:))) < 1e-8);
        end
    end

    function test_info_cc_slice()
        cases = { ...
            struct('ntrl', 18, 'npage', 9, 'xdim', 1, 'ydim', 1), ...
            struct('ntrl', 20, 'npage', 7, 'xdim', 2, 'ydim', 3), ...
            struct('ntrl', 24, 'npage', 5, 'xdim', 4, 'ydim', 4)};
        for ci = 1:numel(cases)
            c = cases{ci};
            x = randn(c.ntrl, c.npage, c.xdim);
            y = randn(c.ntrl, c.ydim);
            expected = mi_gg_vec(x, y, false, true);
            nativeX = permute(x, [1 3 2]);
            actual = info_cc_slice_cpp(nativeX, c.xdim, y, c.ntrl, 2);
            assert(max(abs(actual(:) - expected(:))) < 1e-12);
            if exist('info_cc_slice_cpp_capi', 'file') == 3
                capi = info_cc_slice_cpp_capi(nativeX, c.xdim, y, c.ntrl, 2);
                assert(max(abs(actual(:) - capi(:))) < 1e-12);
            end
            if exist('info_cc_slice_nobc_omp', 'file') == 3
                legacy = info_cc_slice_nobc_omp(nativeX, c.xdim, y, c.ntrl, 2);
                assert(max(abs(actual(:) - legacy(:))) < 1e-12);
            end
        end
    end

    function test_info_cc_single_page()
        ntrl = 20;
        xdim = 2;
        ydim = 3;
        x = randn(ntrl, 1, xdim);
        y = randn(ntrl, ydim);
        expected = mi_gg_vec(x, y, false, true);
        nativeX = permute(x, [1 3 2]);
        singlePageX = reshape(nativeX, [ntrl, xdim]);
        actual = info_cc_slice_cpp(singlePageX, xdim, y, ntrl, 2);
        assert(max(abs(actual(:) - expected(:))) < 1e-12);

        if exist('info_cc_slice_cpp_capi', 'file') == 3
            capi = info_cc_slice_cpp_capi(singlePageX, xdim, y, ntrl, 2);
            assert(max(abs(actual(:) - capi(:))) < 1e-12);
            xBad = randn(2, 1);
            yBad = randn(2, 1);
            assert_error_contains(@() info_cc_slice_cpp_capi(xBad, 1, yBad, 2, 2), ...
                'info_cc_slice_cpp requires Ntrl > Xdim + Ydim');
        end
    end

    function test_info_cd_slice()
        x = randn(2, 24, 5);
        y = int32(mod((0:23)', 4));
        expected = mi_model_gd_vec(permute(x, [2 3 1]), y, 4, false, false);
        actual = info_cd_slice_cpp(x, 2, y, 4, 24, 2);
        assert(max(abs(actual(:) - expected(:))) < 1e-12);
        if exist('info_cd_slice_nobc_omp', 'file') == 3
            legacy = info_cd_slice_nobc_omp(x, 2, int16(double(y) + 1), 4, 24, 2);
            assert(max(abs(actual(:) - legacy(:))) < 1e-12);
        end
    end

    function test_info_cd_single_page()
        x = randn(2, 24, 1);
        y = int32(mod((0:23)', 4));
        expected = mi_model_gd_vec(permute(x, [2 3 1]), y, 4, false, false);
        singlePageX = reshape(x, [2, 24]);
        actual = info_cd_slice_cpp(singlePageX, 2, y, 4, 24, 2);
        assert(max(abs(actual(:) - expected(:))) < 1e-12);
    end

    function assert_error_contains(fun, messageFragment)
        didError = false;
        try
            fun();
        catch ME
            didError = true;
            if isempty(strfind(ME.message, messageFragment))
                error('Unexpected error message: %s', ME.message);
            end
        end
        if ~didError
            error('Expected error containing "%s" was not raised', messageFragment);
        end
    end
end
