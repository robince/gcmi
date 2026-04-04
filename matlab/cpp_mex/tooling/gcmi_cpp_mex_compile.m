function artifacts = gcmi_cpp_mex_compile(varargin)
p = inputParser;
p.addParameter('Targets', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('Verbose', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;

cfg = gcmi_cpp_mex_config();
if ischar(opts.Targets) || isstring(opts.Targets)
    opts.Targets = cellstr(opts.Targets);
end

targets = target_specs(cfg);
if isempty(opts.Targets)
    requested = {targets.name};
else
    requested = cellfun(@char, opts.Targets, 'UniformOutput', false);
end

if ~exist(cfg.BuildDir, 'dir')
    mkdir(cfg.BuildDir);
end
if ~exist(cfg.OutputDir, 'dir')
    mkdir(cfg.OutputDir);
end

artifacts = cell(1, numel(requested));
for i = 1:numel(requested)
    idx = find(strcmp(requested{i}, {targets.name}), 1);
    if isempty(idx)
        error('gcmi_cpp_mex_compile: unknown target %s', requested{i})
    end
    spec = targets(idx);
    if spec.classic_c_wrapper
        build_classic_c_wrapper_target(cfg, spec, opts.Verbose);
    else
        mexArgs = {
            '-outdir', cfg.OutputDir, ...
            '-output', spec.name, ...
            spec.api, ...
            ['-DGCMI_CPP_MATLAB_RELEASE=\"' cfg.Release '\"'], ...
            ['-DGCMI_CPP_ARCH=\"' cfg.Arch '\"'], ...
            ['-DGCMI_CPP_MEXEXT=\"' cfg.MexExt '\"'], ...
            ['-I' cfg.IncludeDir], ...
            ['-I' cfg.SourceDir]};
        if spec.use_openmp
            if ~isempty(cfg.OmpIncludeDir)
                mexArgs{end+1} = ['-I' cfg.OmpIncludeDir];
            end
            mexArgs{end+1} = [cfg.CxxStdFlag cfg.OmpCxxFlag];
            for k_ = 1:numel(cfg.OmpLinkFlags)
                mexArgs{end+1} = cfg.OmpLinkFlags{k_};
            end
        else
            mexArgs{end+1} = cfg.CxxStdFlag;
        end
        if opts.Verbose
            mexArgs{end+1} = '-v';
        end
        mexArgs = [mexArgs, spec.sources, spec.libs]; %#ok<AGROW>
        mex(mexArgs{:});
    end
    artifacts{i} = fullfile(cfg.OutputDir, [spec.name '.' cfg.MexExt]);
end

buildInfo = struct( ...
    'matlab_release', cfg.Release, ...
    'arch', cfg.Arch, ...
    'mexext', cfg.MexExt, ...
    'compiler', cfg.CompilerName, ...
    'compiler_version', cfg.CompilerVersion, ...
    'matlab_root', cfg.MatlabRoot, ...
    'matlab_bin_dir', cfg.MatlabBinDir, ...
    'omp_include_dir', cfg.OmpIncludeDir, ...
    'openmp_runtime', cfg.OpenMPRuntime, ...
    'blas_library', cfg.BlasLibrary, ...
    'lapack_library', cfg.LapackLibrary, ...
    'targets', {requested}, ...
    'generated_at', char(datetime('now', TimeZone='local', Format='yyyy-MM-dd''T''HH:mm:ssXXX')));
buildInfoPath = fullfile(cfg.OutputDir, 'build-info.json');
[fid, msg] = fopen(buildInfoPath, 'w');
if fid < 0
    error('gcmi_cpp_mex_compile:BuildInfoOpenFailed', ...
        'Unable to open "%s" for writing: %s', buildInfoPath, msg);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(buildInfo, PrettyPrint=true));
end

function targets = target_specs(cfg)
targets = struct( ...
    'name', {}, ...
    'sources', {}, ...
    'object_sources', {}, ...
    'libs', {}, ...
    'use_openmp', {}, ...
    'api', {}, ...
    'classic_c_wrapper', {});

targets(end+1) = make_target(cfg, 'gcmi_cpp_ping', {'gcmi_cpp_ping.cpp'}, {}, false);
targets(end+1) = make_target(cfg, 'gcmi_cpp_blas_probe', {'gcmi_cpp_blas_probe.cpp'}, {'-lmwblas', '-lmwlapack'}, false);
targets(end+1) = make_target(cfg, 'gcmi_cpp_omp_probe', {'gcmi_cpp_omp_probe.cpp'}, {}, true);
targets(end+1) = make_target(cfg, 'gcmi_cpp_runtime_probe', {'gcmi_cpp_runtime_probe.cpp'}, {'-lmwblas', '-lmwlapack'}, true);
targets(end+1) = make_target(cfg, 'copnorm_slice_cpp', {'copnorm_slice_cpp.cpp', 'gcmi_kernels.cpp'}, {}, true);
targets(end+1) = make_target(cfg, 'info_cc_slice_cpp', {'info_cc_slice_cpp.cpp', 'gcmi_kernels.cpp'}, {'-lmwblas', '-lmwlapack'}, true);
targets(end+1) = make_target(cfg, 'info_cc_slice_cpp_capi', {'info_cc_slice_cpp_capi.c'}, {'-lmwblas', '-lmwlapack'}, true, '-R2017b', true, {'gcmi_kernels.cpp', 'info_cc_slice_bridge.cpp'});
targets(end+1) = make_target(cfg, 'info_cd_slice_cpp', {'info_cd_slice_cpp.cpp', 'gcmi_kernels.cpp'}, {'-lmwblas', '-lmwlapack'}, true);
end

function target = make_target(cfg, name, mexSources, libs, useOpenMP, api, classicCWrapper, objectSources)
if nargin < 6
    api = '-R2018a';
end
if nargin < 7
    classicCWrapper = false;
end
if nargin < 8
    objectSources = {};
end
target = struct();
target.name = name;
target.sources = cellfun(@(x) fullfile(cfg.MexDir, x), mexSources, 'UniformOutput', false);
target.object_sources = cellfun(@(x) i_resolve_source(cfg, x), objectSources, 'UniformOutput', false);
target.libs = libs;
if any(strcmp('gcmi_kernels.cpp', mexSources))
    target.sources = cellfun(@(x) i_resolve_source(cfg, x), mexSources, 'UniformOutput', false);
end
target.use_openmp = useOpenMP;
target.api = api;
target.classic_c_wrapper = classicCWrapper;
end

function build_classic_c_wrapper_target(cfg, spec, verbose)
objectPaths = cell(1, numel(spec.object_sources));
for i = 1:numel(spec.object_sources)
    [~, baseName] = fileparts(spec.object_sources{i});
    objectPaths{i} = fullfile(cfg.BuildDir, [baseName cfg.ObjExt]);
    compileArgs = {
        '-c', ...
        '-outdir', cfg.BuildDir, ...
        spec.api, ...
        ['-DGCMI_CPP_MATLAB_RELEASE=\"' cfg.Release '\"'], ...
        ['-DGCMI_CPP_ARCH=\"' cfg.Arch '\"'], ...
        ['-DGCMI_CPP_MEXEXT=\"' cfg.MexExt '\"'], ...
        ['-I' cfg.IncludeDir], ...
        ['-I' cfg.SourceDir]};
    if spec.use_openmp && ~isempty(cfg.OmpIncludeDir)
        compileArgs{end+1} = ['-I' cfg.OmpIncludeDir];
    end
    if spec.use_openmp
        compileArgs{end+1} = [cfg.CxxStdFlag cfg.OmpCxxFlag];
    else
        compileArgs{end+1} = cfg.CxxStdFlag;
    end
    compileArgs{end+1} = spec.object_sources{i};
    if verbose
        compileArgs{end+1} = '-v';
    end
    mex(compileArgs{:});
end

linkArgs = {
    '-outdir', cfg.OutputDir, ...
    '-output', spec.name, ...
    spec.api, ...
    ['-DGCMI_CPP_MATLAB_RELEASE=\"' cfg.Release '\"'], ...
    ['-DGCMI_CPP_ARCH=\"' cfg.Arch '\"'], ...
    ['-DGCMI_CPP_MEXEXT=\"' cfg.MexExt '\"'], ...
    ['-I' cfg.IncludeDir], ...
    ['-I' cfg.SourceDir]};
if spec.use_openmp
    linkFlags = cfg.OmpClassicLinkFlags;
else
    linkFlags = {};
end
for k_ = 1:numel(linkFlags)
    linkArgs{end+1} = linkFlags{k_}; %#ok<AGROW>
end
if verbose
    linkArgs{end+1} = '-v';
end
linkArgs = [linkArgs, spec.sources, objectPaths, spec.libs]; %#ok<AGROW>
mex(linkArgs{:});
end

function pathName = i_resolve_source(cfg, name)
if strcmp(name, 'gcmi_kernels.cpp') || strcmp(name, 'info_cc_slice_bridge.cpp')
    pathName = fullfile(cfg.SourceDir, name);
else
    pathName = fullfile(cfg.MexDir, name);
end
end
