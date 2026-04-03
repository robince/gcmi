function cfg = gcmi_cpp_mex_config()
toolingDir = fileparts(mfilename('fullpath'));
cppMexDir = fileparts(toolingDir);
matlabDir = fileparts(cppMexDir);
repoRoot = fileparts(matlabDir);

release = version('-release');
arch = computer('arch');
matlabRootDir = matlabroot;
matlabBinDir = fullfile(matlabRootDir, 'bin', arch);

cxx = mex.getCompilerConfigurations('C++', 'Selected');
if isempty(cxx)
    error('gcmi_cpp_mex_config: no MATLAB C++ compiler selected; run mex -setup C++')
end

compilerFamily = i_compiler_family(cxx);
[cxxStdFlag, ompCxxFlag, ompLinkFlags, ompClassicLinkFlags, objExt] = ...
    i_compiler_flags(compilerFamily, matlabBinDir);
ompIncludeDir = i_find_omp_include(matlabRootDir, arch);
libExt = i_lib_extension();

compilerTag = regexprep(lower(cxx.Name), '[^a-z0-9]+', '_');
cfg = struct();
cfg.RepoRoot            = repoRoot;
cfg.MatlabDir           = matlabDir;
cfg.CppMexDir           = cppMexDir;
cfg.IncludeDir          = fullfile(cppMexDir, 'include');
cfg.SourceDir           = fullfile(cppMexDir, 'src');
cfg.MexDir              = fullfile(cppMexDir, 'mex');
cfg.TestsDir            = fullfile(cppMexDir, 'tests');
cfg.ToolingDir          = toolingDir;
cfg.OutputDir           = fullfile(cppMexDir, 'bin', release, mexext);
cfg.BuildDir            = fullfile(repoRoot, 'build', 'matlab', 'cpp_mex', release, arch, compilerTag);
cfg.Release             = release;
cfg.Arch                = arch;
cfg.MexExt              = mexext;
cfg.MatlabRoot          = matlabRootDir;
cfg.MatlabBinDir        = matlabBinDir;
cfg.OmpIncludeDir       = ompIncludeDir;
cfg.CompilerName        = cxx.Name;
cfg.CompilerVersion     = cxx.Version;
cfg.CompilerFamily      = compilerFamily;
cfg.CxxStdFlag          = cxxStdFlag;
cfg.OmpCxxFlag          = ompCxxFlag;
cfg.OmpLinkFlags        = ompLinkFlags;
cfg.OmpClassicLinkFlags = ompClassicLinkFlags;
cfg.ObjExt              = objExt;
cfg.OpenMPRuntime       = fullfile(matlabBinDir, ['libomp' libExt]);
cfg.BlasLibrary         = fullfile(matlabBinDir, ['libmwblas' libExt]);
cfg.LapackLibrary       = fullfile(matlabBinDir, ['libmwlapack' libExt]);
end

% ---------------------------------------------------------------------------
% Internal helpers
% ---------------------------------------------------------------------------

function family = i_compiler_family(cxx)
% Classify the selected C++ compiler into a small set of flag families.
% Keying on the compiler name rather than the OS means we handle MSVC
% correctly on Windows even if MinGW is also installed, and is robust to
% non-default toolchain choices on any platform.
name = lower(cxx.Name);
if ~isempty(regexpi(name, 'microsoft visual c\+\+|msvc', 'once'))
    family = 'msvc';
elseif ~isempty(regexpi(name, 'mingw', 'once'))
    family = 'gcc';
elseif ~isempty(regexpi(name, 'clang|apple', 'once'))
    family = 'clang';
elseif ~isempty(regexpi(name, 'gnu|g\+\+', 'once'))
    family = 'gcc';
else
    % Unknown compiler: assume GCC-compatible flags as a conservative default.
    family = 'gcc';
end
end

function libExt = i_lib_extension()
% Shared-library extension is an OS-level fact, not a compiler choice.
if ismac
    libExt = '.dylib';
elseif ispc
    libExt = '.dll';
else
    libExt = '.so';
end
end

function [cxxStdFlag, ompCxxFlag, ompLinkFlags, ompClassicLinkFlags, objExt] = ...
        i_compiler_flags(family, matlabBinDir)
switch family
    case 'msvc'
        % MSVC: /std:c++17 goes through COMPFLAGS; /openmp activates the
        % built-in vcomp runtime — no extra link libraries needed.
        cxxStdFlag          = 'COMPFLAGS=$COMPFLAGS /std:c++17';
        ompCxxFlag          = ' /openmp';
        ompLinkFlags        = {};
        ompClassicLinkFlags = {};
        objExt              = '.obj';
    case 'gcc'
        % GCC (Linux and MinGW-w64 on Windows): -fopenmp covers both the
        % compile-time header and the libgomp runtime link.
        % On Linux add rpath so MATLAB's own shared libs are found; on
        % Windows DLL search uses PATH so rpath is omitted.
        if ispc
            rpathFlag = '';
        else
            rpathFlag = [' -Wl,-rpath,' matlabBinDir];
        end
        cxxStdFlag          = 'CXXFLAGS=$CXXFLAGS -std=c++17';
        ompCxxFlag          = ' -fopenmp';
        ompLinkFlags        = {['LDFLAGS=$LDFLAGS -fopenmp' rpathFlag]};
        ompClassicLinkFlags = { ...
            ['LDFLAGS=$LDFLAGS -fopenmp' rpathFlag], ...
            'LINKLIBS=$LINKLIBS -lstdc++'};
        objExt              = '.o';
    otherwise
        % Apple Clang (and any unrecognised Clang variant): -Xpreprocessor
        % passes -fopenmp through the driver to clang-cc1.  Link against
        % MATLAB-bundled libomp and set rpath so the dylib is found at
        % runtime without DYLD_LIBRARY_PATH.
        cxxStdFlag          = 'CXXFLAGS=$CXXFLAGS -std=c++17';
        ompCxxFlag          = ' -Xpreprocessor -fopenmp';
        ompLinkFlags        = { ...
            ['LDFLAGS=$LDFLAGS -Wl,-rpath,' matlabBinDir], ...
            ['LINKLIBS=$LINKLIBS -L' matlabBinDir ' -lomp']};
        ompClassicLinkFlags = { ...
            ['LDFLAGS=$LDFLAGS -Wl,-rpath,' matlabBinDir], ...
            ['LINKLIBS=$LINKLIBS -L' matlabBinDir ' -lomp -lc++']};
        objExt              = '.o';
end
end

function ompIncludeDir = i_find_omp_include(matlabRootDir, arch)
% omp.h is not bundled alongside Apple Clang, so search for the copy
% shipped with MATLAB.  GCC and MSVC include omp.h in their own toolchain
% directories, so no extra search is needed on Linux or Windows.
ompIncludeDir = '';
if ~ismac
    return
end
candidates = { ...
    fullfile(matlabRootDir, 'toolbox', 'eml', 'externalDependency', 'omp', arch, 'include'), ...
    fullfile(matlabRootDir, 'toolbox', 'coder', 'clang_api', 'llvm-include', arch)};
for i = 1:numel(candidates)
    if exist(fullfile(candidates{i}, 'omp.h'), 'file')
        ompIncludeDir = candidates{i};
        return
    end
end
warning('gcmi_cpp_mex_config:ompNotFound', ...
    'Could not find MATLAB-shipped omp.h; OpenMP targets may fail to build.');
end
