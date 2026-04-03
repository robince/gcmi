function addedPaths = setup_gcmi(varargin)
%SETUP_GCMI Add the GCMI MATLAB package to the MATLAB path.
%   setup_gcmi adds the matlab/ directory from this repository to the path.
%   If a matching native C++ MEX runtime exists under matlab/cpp_mex/bin/,
%   setup_gcmi also adds that directory automatically.
%   setup_gcmi('IncludeExamples', true) also adds matlab_examples/.
%   setup_gcmi('SavePath', true) persists the path with SAVEPATH.
%
%   addedPaths = setup_gcmi(...) returns the directories added during the
%   call. The helper is designed to be called from the repository root after
%   the root itself has been added to the MATLAB path.

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'IncludeExamples', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'IncludeNative', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'SavePath', false, @(x) islogical(x) && isscalar(x));
parse(p, varargin{:});
opts = p.Results;

repoRoot = fileparts(mfilename('fullpath'));
matlabDir = fullfile(repoRoot, 'matlab');
exampleDir = fullfile(repoRoot, 'matlab_examples');
nativeDir = fullfile(repoRoot, 'matlab', 'cpp_mex', 'bin', version('-release'), mexext);

if ~exist(matlabDir, 'dir')
    error('setup_gcmi: matlab directory not found at %s', matlabDir)
end

addedPaths = {matlabDir};
addpath(matlabDir);

if opts.IncludeNative && exist(nativeDir, 'dir')
    addpath(nativeDir);
    addedPaths{end+1} = nativeDir;
end

if opts.IncludeExamples
    if exist(exampleDir, 'dir')
        addpath(exampleDir);
        addedPaths{end+1} = exampleDir;
    else
        warning('setup_gcmi: example directory not found at %s', exampleDir)
    end
end

if opts.SavePath
    status = savepath;
    if status ~= 0
        warning('setup_gcmi: could not persist the MATLAB path')
    end
end

if nargout == 0
    clear addedPaths
end
