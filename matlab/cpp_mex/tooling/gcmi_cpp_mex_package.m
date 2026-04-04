function archivePath = gcmi_cpp_mex_package(varargin)
cfg = gcmi_cpp_mex_config();
gcmi_cpp_mex_compile(varargin{:});
packageDir = fullfile(cfg.RepoRoot, 'build', 'matlab', 'packages');
if ~exist(packageDir, 'dir')
    mkdir(packageDir);
end
archivePath = fullfile(packageDir, sprintf('gcmi_cpp_mex_%s_%s.zip', cfg.Release, cfg.Arch));
if exist(archivePath, 'file')
    delete(archivePath);
end
files = dir(fullfile(cfg.OutputDir, '*'));
files = files(~[files.isdir]);
zipInputs = fullfile({files.folder}, {files.name});
zip(archivePath, zipInputs, cfg.OutputDir);
fprintf('%s\n', archivePath);
end
