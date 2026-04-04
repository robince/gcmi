function output_dir = run_matlab_benchmarks(varargin)
% RUN_MATLAB_BENCHMARKS Run shared-contract benchmarks for MATLAB and MEX backends.
%
% Example:
%   run_matlab_benchmarks('FixtureIds', {'copnorm_medium_f64'}, 'ThreadCounts', [1 4 8]);
%
% To benchmark a future C++ MEX implementation with different entrypoint names:
%   fmap = struct( ...
%       'copnorm_slice', 'copnorm_slice_cpp', ...
%       'info_cc_slice', 'info_cc_slice_cpp');
%   run_matlab_benchmarks('OptimizedLabel', 'cpp_mex', 'OptimizedFunctions', fmap);

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

opts = parse_inputs(root, varargin{:});
add_optional_paths(opts.LegacyPaths);
add_optional_paths(opts.OptimizedPaths);
fixtures = load_fixtures(fullfile(root, 'benchmarks', 'fixtures_manifest.json'), opts.FixtureIds);
run_id = make_run_id(root);
output_dir = fullfile(opts.OutputRoot, run_id);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

results = {};
for fi = 1:numel(fixtures)
    fixture = fixtures(fi);
    data = fixture_data(fixture);
    data = prepare_benchmark_data(fixture, data, opts.OptimizedLabelEncoding);

    ref_record = measure_fixture(fixture, data, 'matlab_reference', 1, opts.Repeat, opts);
    ref_record.run_id = run_id;
    ref_record.git_revision = git_revision(root);
    results{end+1} = ref_record; %#ok<AGROW>

    for ti = 1:numel(opts.ThreadCounts)
        record = measure_fixture(fixture, data, opts.OptimizedLabel, opts.ThreadCounts(ti), opts.Repeat, opts);
        record.run_id = run_id;
        record.git_revision = git_revision(root);
        results{end+1} = record; %#ok<AGROW>
    end
end

results = attach_relative_metrics(results);
write_environment_json(fullfile(output_dir, 'environment.json'), build_environment(run_id, opts));
write_results_jsonl(fullfile(output_dir, 'results.jsonl'), results);

fprintf('%s\n', output_dir);
end

function opts = parse_inputs(root, varargin)
p = inputParser;
p.addParameter('FixtureIds', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('ThreadCounts', default_thread_counts(), @isnumeric);
p.addParameter('Repeat', 10, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('OutputRoot', fullfile(root, 'benchmarks', 'runs'), @(x) ischar(x) || isstring(x));
p.addParameter('Notes', '', @(x) ischar(x) || isstring(x));
p.addParameter('OptimizedLabel', 'matlab_mex', @(x) ischar(x) || isstring(x));
p.addParameter('OptimizedFunctions', default_optimized_functions(), @isstruct);
p.addParameter('LegacyPaths', fullfile(root, 'extern', 'gcmi_mex'), @(x) ischar(x) || isstring(x) || iscell(x));
p.addParameter('OptimizedPaths', {}, @(x) ischar(x) || isstring(x) || iscell(x));
p.addParameter('OptimizedLabelEncoding', 'legacy_one_based', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

opts = p.Results;
opts.OutputRoot = char(opts.OutputRoot);
opts.Notes = char(opts.Notes);
opts.OptimizedLabel = char(opts.OptimizedLabel);
opts.LegacyPaths = normalize_paths(opts.LegacyPaths);
opts.OptimizedPaths = normalize_paths(opts.OptimizedPaths);
opts.OptimizedLabelEncoding = char(opts.OptimizedLabelEncoding);
if ischar(opts.FixtureIds) || isstring(opts.FixtureIds)
    opts.FixtureIds = cellstr(opts.FixtureIds);
end
opts.ThreadCounts = unique(double(opts.ThreadCounts(:)'));
end

function paths = normalize_paths(value)
if isempty(value)
    paths = {};
elseif ischar(value) || isstring(value)
    paths = cellstr(value);
elseif iscell(value)
    paths = cellfun(@char, value, 'UniformOutput', false);
else
    error('run_matlab_benchmarks: invalid path input')
end
paths = paths(~cellfun(@isempty, paths));
end

function add_optional_paths(paths)
for i = 1:numel(paths)
    if exist(paths{i}, 'dir')
        addpath(paths{i});
    end
end
end

function counts = default_thread_counts()
n = logical_cores();
counts = unique([1 2 4 8 n]);
counts = counts(counts >= 1 & counts <= n);
if isempty(counts)
    counts = 1;
end
end

function fmap = default_optimized_functions()
fmap = struct( ...
    'copnorm_slice', 'copnorm_slice_omp_c_double', ...
    'info_cc_slice', 'info_cc_slice_nobc_omp', ...
    'info_cc_multi', 'info_cc_multi_nobc_omp', ...
    'info_cc_slice_indexed', 'info_cc_slice_indexed_nobc_omp', ...
    'info_c1d_slice', 'info_c1d_slice_nobc_omp', ...
    'info_cd_slice', 'info_cd_slice_nobc_omp', ...
    'info_dc_slice_bc', 'info_dc_slice_bc_omp');
end

function fixtures = load_fixtures(pathname, requested_ids)
payload = jsondecode(fileread(pathname));
fixtures = normalize_fixture_array(payload.fixtures);
if isempty(requested_ids)
    return
end
if isstring(requested_ids)
    requested_ids = cellstr(requested_ids);
end
requested_ids = cellfun(@char, requested_ids, 'UniformOutput', false);
keep = false(1, numel(fixtures));
for i = 1:numel(fixtures)
    keep(i) = any(strcmp(fixtures(i).fixture_id, requested_ids));
end
missing = requested_ids(~ismember(requested_ids, {fixtures.fixture_id}));
if ~isempty(missing)
    error('run_matlab_benchmarks:UnknownFixture', 'Unknown fixture ids: %s', strjoin(missing, ', '));
end
fixtures = fixtures(keep);
end

function fixtures = normalize_fixture_array(raw_fixtures)
field_names = {'fixture_id', 'kernel', 'dtype', 'ntrl', 'npage', ...
    'xdim', 'ydim', 'ym', 'xm', 'seed'};

if isstruct(raw_fixtures)
    raw_cells = num2cell(raw_fixtures);
elseif iscell(raw_fixtures)
    raw_cells = raw_fixtures;
else
    error('run_matlab_benchmarks:InvalidFixturePayload', ...
        'fixtures_manifest.json did not decode into a supported fixtures container');
end

if isempty(raw_cells)
    fixtures = empty_fixture_struct();
    return
end

fixtures = repmat(empty_fixture_struct(), 1, numel(raw_cells));
for i = 1:numel(raw_cells)
    item = raw_cells{i};
    if ~isstruct(item)
        error('run_matlab_benchmarks:InvalidFixturePayload', ...
            'Each decoded fixture must be a struct');
    end
    for fi = 1:numel(field_names)
        name = field_names{fi};
        if isfield(item, name)
            fixtures(i).(name) = item.(name);
        else
            fixtures(i).(name) = [];
        end
    end
end
end

function fixture = empty_fixture_struct()
fixture = struct( ...
    'fixture_id', '', ...
    'kernel', '', ...
    'dtype', '', ...
    'ntrl', [], ...
    'npage', [], ...
    'xdim', [], ...
    'ydim', [], ...
    'ym', [], ...
    'xm', [], ...
    'seed', []);
end

function data = fixture_data(fixture)
rng(double(fixture.seed), 'twister');
dtype = matlab_float_class(fixture.dtype);
switch fixture.kernel
    case 'copnorm_slice'
        data.X = cast(rand(fixture.ntrl, fixture.npage), dtype);
    case 'info_cc_slice'
        data.X = cast(randn(fixture.ntrl, fixture.npage, fixture.xdim), dtype);
        data.Y = cast(randn(fixture.ntrl, fixture.ydim), dtype);
    case 'info_cc_multi'
        data.X = cast(randn(fixture.ntrl, fixture.npage, fixture.xdim), dtype);
        data.Y = cast(randn(fixture.ntrl, fixture.npage, fixture.ydim), dtype);
    case 'info_cc_slice_indexed'
        data.X = cast(randn(fixture.ntrl, fixture.npage, fixture.xdim), dtype);
        data.Y = cast(randn(fixture.ntrl, fixture.ydim), dtype);
        data.Xidx = int32(randi([0, fixture.npage - 1], fixture.npage, 1));
    case 'info_c1d_slice'
        data.X = cast(randn(fixture.ntrl, fixture.npage), dtype);
        data.Y = balanced_labels(fixture.ntrl, fixture.ym);
    case 'info_cd_slice'
        data.X = cast(randn(fixture.ntrl, fixture.npage, fixture.xdim), dtype);
        data.Y = balanced_labels(fixture.ntrl, fixture.ym);
    case 'info_dc_slice_bc'
        data.Y = cast(randn(fixture.ntrl, fixture.ydim), dtype);
        data.X = zeros(fixture.ntrl, fixture.npage, 'int32');
        for page = 1:fixture.npage
            data.X(:, page) = balanced_labels(fixture.ntrl, fixture.xm);
        end
    otherwise
        error('run_matlab_benchmarks:UnsupportedKernel', 'Unsupported kernel: %s', fixture.kernel);
end
end

function data = prepare_benchmark_data(fixture, data, optimizedLabelEncoding)
switch fixture.kernel
    case 'info_cc_slice'
        data.optimized_X = permute(data.X, [1 3 2]);
        data.optimized_bias = biasterms_cc(fixture.xdim, fixture.ydim, fixture.ntrl);
    case 'info_cc_multi'
        data.optimized_X = permute(data.X, [1 3 2]);
        data.optimized_Y = permute(data.Y, [1 3 2]);
        data.optimized_bias = biasterms_cc(fixture.xdim, fixture.ydim, fixture.ntrl);
    case 'info_cc_slice_indexed'
        data.optimized_X = permute(data.X, [1 3 2]);
        data.optimized_Xidx = int32(double(data.Xidx) + 1);
        data.optimized_bias = biasterms_cc(fixture.xdim, fixture.ydim, fixture.ntrl);
    case 'info_c1d_slice'
        data.optimized_labels = encode_discrete_labels(data.Y, optimizedLabelEncoding);
        data.optimized_bincount = histcounts(double(data.Y), -0.5:1:(fixture.ym - 0.5));
        data.optimized_bias = biasterms_cd(1, fixture.ntrl, data.optimized_bincount);
    case 'info_cd_slice'
        data.optimized_X = permute(data.X, [3 1 2]);
        data.optimized_labels = encode_discrete_labels(data.Y, optimizedLabelEncoding);
        data.optimized_bincount = histcounts(double(data.Y), -0.5:1:(fixture.ym - 0.5));
        data.optimized_bias = biasterms_cd(fixture.xdim, fixture.ntrl, data.optimized_bincount);
    case 'info_dc_slice_bc'
        data.optimized_X = encode_discrete_labels(data.X, optimizedLabelEncoding);
end
end

function dtype = matlab_float_class(name)
name = char(name);
switch name
    case 'float64'
        dtype = 'double';
    case 'float32'
        dtype = 'single';
    otherwise
        error('run_matlab_benchmarks:UnsupportedDType', ...
            'Unsupported fixture dtype: %s', name);
end
end

function labels = balanced_labels(ntrl, nclasses)
repeats = ceil(ntrl / nclasses);
labels = repmat(int32(0:nclasses-1), 1, repeats);
labels = labels(1:ntrl);
labels = labels(randperm(ntrl)).';
end

function record = measure_fixture(fixture, data, implementation, thread_count, repeat, opts)
f = @() dispatch_kernel(fixture, data, implementation, thread_count, opts);
tic;
f();
first_ms = toc * 1000.0;

times_ms = zeros(repeat, 1);
for i = 1:repeat
    tic;
    f();
    times_ms(i) = toc * 1000.0;
end

steady_ms = median(times_ms);
compile_ms = [];
if ~strcmp(implementation, 'matlab_reference')
    compile_ms = max(first_ms - steady_ms, 0.0);
end

record = base_record(fixture);
record.language = 'matlab';
record.implementation = implementation;
record.thread_count = thread_count;
record.compile_time_ms = compile_ms;
record.steady_state_time_ms = steady_ms;
record.p10_time_ms = percentile(times_ms, 10);
record.p90_time_ms = percentile(times_ms, 90);
record.slices_per_second = fixture.npage / (steady_ms / 1000.0);
record.speedup_vs_reference = [];
record.speedup_vs_1thread = [];
record.scaling_efficiency = [];
record.run_id = '';
record.git_revision = '';
record.notes = [];
end

function record = base_record(fixture)
record = struct();
record.kernel = fixture.kernel;
record.language = '';
record.implementation = '';
record.dtype = fixture.dtype;
record.thread_count = 1;
record.ntrl = fixture.ntrl;
record.npage = fixture.npage;
record.xdim = field_or_null(fixture, 'xdim');
record.ydim = field_or_null(fixture, 'ydim');
record.ym = field_or_null(fixture, 'ym');
record.xm = field_or_null(fixture, 'xm');
record.compile_time_ms = [];
record.steady_state_time_ms = [];
record.p10_time_ms = [];
record.p90_time_ms = [];
record.slices_per_second = [];
record.speedup_vs_reference = [];
record.speedup_vs_1thread = [];
record.scaling_efficiency = [];
record.fixture_id = fixture.fixture_id;
record.run_id = '';
record.git_revision = '';
record.notes = [];
end

function value = field_or_null(s, name)
if isfield(s, name)
    value = s.(name);
else
    value = [];
end
end

function out = dispatch_kernel(fixture, data, implementation, thread_count, opts)
switch implementation
    case 'matlab_reference'
        out = dispatch_reference(fixture, data);
    otherwise
        out = dispatch_optimized(fixture, data, thread_count, opts.OptimizedFunctions, opts.OptimizedLabelEncoding);
end
end

function out = dispatch_reference(fixture, data)
switch fixture.kernel
    case 'copnorm_slice'
        out = zeros(size(data.X), 'like', data.X);
        for page = 1:size(data.X, 2)
            out(:, page) = cast(copnorm(data.X(:, page)), class(data.X));
        end
    case 'info_cc_slice'
        out = mi_gg_vec(data.X, data.Y, true, false);
    case 'info_cc_multi'
        out = zeros(size(data.X, 2), 1);
        for page = 1:size(data.X, 2)
            out(page) = mi_gg(squeeze(data.X(:, page, :)), squeeze(data.Y(:, page, :)), true, false);
        end
    case 'info_cc_slice_indexed'
        out = mi_gg_vec(data.X(:, double(data.Xidx) + 1, :), data.Y, true, false);
    case 'info_c1d_slice'
        out = mi_model_gd_vec(reshape(data.X, size(data.X, 1), size(data.X, 2), 1), data.Y, fixture.ym, true, false);
    case 'info_cd_slice'
        out = mi_model_gd_vec(data.X, data.Y, fixture.ym, true, false);
    case 'info_dc_slice_bc'
        out = zeros(size(data.X, 2), 1);
        for page = 1:size(data.X, 2)
            out(page) = mi_model_dg_local(data.X(:, page), data.Y, fixture.xm, true);
        end
    otherwise
        error('run_matlab_benchmarks:UnsupportedKernel', 'Unsupported kernel: %s', fixture.kernel);
end
end

function out = dispatch_optimized(fixture, data, thread_count, fmap, encoding)
fname = mex_function_name(fmap, fixture.kernel);
switch fixture.kernel
    case 'copnorm_slice'
        out = feval(fname, data.X, thread_count);
    case 'info_cc_slice'
        out = feval(fname, data.optimized_X, fixture.xdim, data.Y, fixture.ntrl, thread_count);
        out = out - data.optimized_bias;
    case 'info_cc_multi'
        out = feval(fname, data.optimized_X, fixture.xdim, data.optimized_Y, fixture.ydim, fixture.ntrl, thread_count);
        out = out - data.optimized_bias;
    case 'info_cc_slice_indexed'
        out = call_info_cc_slice_indexed(fname, data.optimized_X, fixture.xdim, data.optimized_Xidx, data.Y, fixture.ydim, fixture.ntrl, thread_count);
        out = out - data.optimized_bias;
    case 'info_c1d_slice'
        out = feval(fname, data.X, data.optimized_labels, fixture.ym, fixture.ntrl, thread_count);
        out = out - data.optimized_bias;
    case 'info_cd_slice'
        out = feval(fname, data.optimized_X, fixture.xdim, data.optimized_labels, fixture.ym, fixture.ntrl, thread_count);
        out = out - data.optimized_bias;
    case 'info_dc_slice_bc'
        out = call_info_dc_slice_bc(fname, data.optimized_X, fixture.xm, data.Y, fixture.ntrl, thread_count);
    otherwise
        error('run_matlab_benchmarks:UnsupportedKernel', 'Unsupported kernel: %s', fixture.kernel);
end
end

function out = call_info_cc_slice_indexed(fname, X, xdim, Xidx, Y, ydim, ntrl, thread_count)
persistent legacy_signature_cache
if isempty(legacy_signature_cache)
    legacy_signature_cache = struct();
end

field_name = matlab.lang.makeValidName(fname);
if isfield(legacy_signature_cache, field_name)
    if legacy_signature_cache.(field_name) == 6
        out = feval(fname, X, xdim, Xidx, Y, ntrl, thread_count);
    else
        out = feval(fname, X, xdim, Xidx, Y, ydim, ntrl, thread_count);
    end
    return
end

try
    out = feval(fname, X, xdim, Xidx, Y, ydim, ntrl, thread_count);
    legacy_signature_cache.(field_name) = 7;
catch ME
    if contains(ME.message, 'takes 6 inputs')
        out = feval(fname, X, xdim, Xidx, Y, ntrl, thread_count);
        legacy_signature_cache.(field_name) = 6;
    else
        rethrow(ME);
    end
end
end

function out = call_info_dc_slice_bc(fname, X, xm, Y, ntrl, thread_count)
persistent dc_transpose_cache
if isempty(dc_transpose_cache)
    dc_transpose_cache = struct();
end

field_name = matlab.lang.makeValidName(fname);
if isfield(dc_transpose_cache, field_name)
    if dc_transpose_cache.(field_name)
        out = feval(fname, X, xm, Y.', ntrl, thread_count);
    else
        out = feval(fname, X, xm, Y, ntrl, thread_count);
    end
    return
end

try
    out = feval(fname, X, xm, Y, ntrl, thread_count);
    dc_transpose_cache.(field_name) = false;
catch ME
    if contains(ME.message, 'Number of trials does not match data')
        out = feval(fname, X, xm, Y.', ntrl, thread_count);
        dc_transpose_cache.(field_name) = true;
    else
        rethrow(ME);
    end
end
end

function labels = encode_discrete_labels(labelsIn, encoding)
switch encoding
    case 'legacy_one_based'
        labels = int16(double(labelsIn) + 1);
    case 'zero_based'
        labels = int32(double(labelsIn));
    otherwise
        error('run_matlab_benchmarks:UnsupportedLabelEncoding', ...
            'Unsupported optimized label encoding: %s', encoding);
end
end

function name = mex_function_name(fmap, kernel)
if ~isfield(fmap, kernel)
    error('run_matlab_benchmarks:MissingOptimizedFunction', ...
        'No optimized function configured for kernel %s', kernel);
end
name = fmap.(kernel);
if isa(name, 'function_handle')
    error('run_matlab_benchmarks:FunctionHandleNotSupported', ...
        'OptimizedFunctions currently expects function names, not handles.');
end
name = char(name);
if isempty(strtrim(name))
    error('run_matlab_benchmarks:MissingOptimizedFunction', ...
        'No optimized entrypoint configured for kernel %s', kernel);
end
if exist(name, 'file') ~= 3 && exist(name, 'file') ~= 2
    error('run_matlab_benchmarks:MissingOptimizedEntrypoint', ...
        'Optimized entrypoint %s was not found on the MATLAB path', name);
end
end

function out = mi_model_dg_local(x, y, Xm, biascorrect)
if nargin < 4
    biascorrect = true;
end

x = x(:);
if size(y, 1) ~= numel(x)
    error('mi_model_dg_local: number of trials do not match');
end

counts = histcounts(double(x), -0.5:1:(Xm - 0.5));
if any(counts == 0)
    error('mi_model_dg_local: empty classes are not supported');
end
ydim = size(y, 2);
if any(counts <= ydim)
    error('mi_model_dg_local: each class needs more than ydim samples');
end

hcond = zeros(Xm, 1);
for xi = 1:Xm
    ym = y(x == (xi - 1), :);
    hcond(xi) = ent_g(ym, biascorrect);
end
hunc = ent_g(y, biascorrect);
w = counts(:) ./ size(y, 1);
out = hunc - sum(w .* hcond);
end

function results = attach_relative_metrics(results)
fixture_ids = cellfun(@(r) r.fixture_id, results, 'UniformOutput', false);
unique_ids = unique(fixture_ids);
for ui = 1:numel(unique_ids)
    idx = find(strcmp(fixture_ids, unique_ids{ui}));
    impls = cellfun(@(r) r.implementation, results(idx), 'UniformOutput', false);
    threads = cellfun(@(r) r.thread_count, results(idx));
    ref_idx = idx(strcmp(impls, 'matlab_reference'));
    opt_idx = idx(~strcmp(impls, 'matlab_reference'));
    opt_one_idx = opt_idx(threads(~strcmp(impls, 'matlab_reference')) == 1);
    if ~isempty(ref_idx)
        ref_ms = results{ref_idx(1)}.steady_state_time_ms;
        for j = idx
            results{j}.speedup_vs_reference = ref_ms / results{j}.steady_state_time_ms;
        end
    end
    if ~isempty(opt_one_idx)
        one_ms = results{opt_one_idx(1)}.steady_state_time_ms;
        for j = idx
            if ~strcmp(results{j}.implementation, 'matlab_reference')
                results{j}.speedup_vs_1thread = one_ms / results{j}.steady_state_time_ms;
                results{j}.scaling_efficiency = results{j}.speedup_vs_1thread / results{j}.thread_count;
            end
        end
    end
end
end

function env = build_environment(run_id, opts)
env = struct();
env.run_id = run_id;
env.language = 'matlab';
env.platform = computer;
env.arch = computer('arch');
env.cpu_model = cpu_model();
env.physical_cores = physical_cores();
env.logical_cores = logical_cores();
env.os_version = os_version();
env.matlab_release = version('-release');
env.python_version = [];
env.numpy_version = [];
env.numba_version = [];
env.llvmlite_version = [];
env.compiler = [];
env.compiler_version = [];
env.openmp_runtime = [];
env.blas_vendor = [];
env.git_revision = git_revision(fileparts(fileparts(mfilename('fullpath'))));
if isempty(opts.Notes)
    env.notes = [];
else
    env.notes = opts.Notes;
end
end

function rev = git_revision(root)
[status, out] = system(sprintf('cd "%s" && git rev-parse --short HEAD', root));
if status == 0
    rev = strtrim(out);
else
    rev = '';
end
end

function run_id = make_run_id(root)
timestamp = datestr(now, 'yyyymmdd-HHMMSS');
rev = git_revision(root);
if isempty(rev)
    rev = 'nogit';
end
run_id = sprintf('%s-matlab-%s-%s', timestamp, computer('arch'), rev);
end

function val = physical_cores()
if ismac
    [status, out] = system('sysctl -n hw.physicalcpu');
    if status == 0
        val = str2double(strtrim(out));
        if ~isnan(val)
            return
        end
    end
elseif isunix
    [status, out] = system('getconf _NPROCESSORS_ONLN');
    if status == 0
        val = str2double(strtrim(out));
        if ~isnan(val)
            return
        end
    end
elseif ispc
    out = getenv('NUMBER_OF_PROCESSORS');
    val = str2double(out);
    if ~isnan(val)
        return
    end
end
val = logical_cores();
end

function val = logical_cores()
if ismac
    [status, out] = system('sysctl -n hw.logicalcpu');
    if status == 0
        val = str2double(strtrim(out));
        if ~isnan(val)
            return
        end
    end
elseif isunix
    [status, out] = system('getconf _NPROCESSORS_ONLN');
    if status == 0
        val = str2double(strtrim(out));
        if ~isnan(val)
            return
        end
    end
elseif ispc
    out = getenv('NUMBER_OF_PROCESSORS');
    val = str2double(out);
    if ~isnan(val)
        return
    end
end
val = 1;
end

function val = cpu_model()
if ismac
    [status, out] = system('sysctl -n machdep.cpu.brand_string');
    if status ~= 0 || isempty(strtrim(out))
        [status, out] = system('sysctl -n hw.model');
    end
elseif isunix
    [status, out] = system('uname -m');
else
    [status, out] = system('wmic cpu get Name /value');
end
if status == 0
    val = strtrim(out);
else
    val = computer;
end
end

function val = os_version()
if ismac
    [status, out] = system('sw_vers -productVersion');
elseif isunix
    [status, out] = system('uname -srv');
else
    [status, out] = system('ver');
end
if status == 0
    val = strtrim(out);
else
    val = '';
end
end

function write_environment_json(pathname, env)
fields = { ...
    'run_id', 'language', 'platform', 'arch', 'cpu_model', ...
    'physical_cores', 'logical_cores', 'os_version', 'matlab_release', ...
    'python_version', 'numpy_version', 'numba_version', 'llvmlite_version', ...
    'compiler', 'compiler_version', 'openmp_runtime', 'blas_vendor', ...
    'git_revision', 'notes'};
    write_text(pathname, encode_json_object(env, fields, true));
end

function write_results_jsonl(pathname, results)
fields = { ...
    'kernel', 'language', 'implementation', 'dtype', 'thread_count', ...
    'ntrl', 'npage', 'xdim', 'ydim', 'ym', 'xm', 'compile_time_ms', ...
    'steady_state_time_ms', 'p10_time_ms', 'p90_time_ms', ...
    'slices_per_second', 'speedup_vs_reference', 'speedup_vs_1thread', ...
    'scaling_efficiency', 'fixture_id', 'run_id', 'git_revision', 'notes'};

fid = fopen(pathname, 'w');
cleanup = onCleanup(@() fclose(fid));
for i = 1:numel(results)
    fprintf(fid, '%s\n', encode_json_object(results{i}, fields, false));
end
end

function out = encode_json_object(s, fields, pretty)
if nargin < 3
    pretty = false;
end
sep = ',';
linebreak = '';
indent = '';
inner_indent = '';
if pretty
    linebreak = sprintf('\n');
    indent = '  ';
    inner_indent = '    ';
end

parts = cell(1, numel(fields));
for i = 1:numel(fields)
    field = fields{i};
    parts{i} = sprintf('%s"%s": %s', inner_indent, field, json_scalar(s.(field)));
end
out = ['{' linebreak strjoin(parts, [sep linebreak]) linebreak indent '}'];
end

function out = json_scalar(val)
if isempty(val)
    out = 'null';
elseif isstring(val)
    if strlength(val) == 0
        out = 'null';
    else
        out = jsonencode(char(val));
    end
elseif ischar(val)
    if isempty(val)
        out = 'null';
    else
        out = jsonencode(val);
    end
elseif isnumeric(val) || islogical(val)
    if isscalar(val)
        if isnan(val)
            out = 'null';
        elseif islogical(val)
            if val
                out = 'true';
            else
                out = 'false';
            end
        elseif abs(val - round(val)) < 1e-12
            out = sprintf('%d', round(val));
        else
            out = sprintf('%.15g', val);
        end
    else
        out = jsonencode(val);
    end
else
    out = jsonencode(val);
end
end

function val = percentile(x, p)
x = sort(x(:));
if isempty(x)
    val = [];
    return
end
if numel(x) == 1
    val = x;
    return
end
pos = 1 + (numel(x) - 1) * (p / 100);
lo = floor(pos);
hi = ceil(pos);
if lo == hi
    val = x(lo);
else
    w = pos - lo;
    val = x(lo) * (1 - w) + x(hi) * w;
end
end

function write_text(pathname, text)
fid = fopen(pathname, 'w');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, text, 'char');
end
