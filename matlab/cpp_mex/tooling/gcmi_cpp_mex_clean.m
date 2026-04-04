function gcmi_cpp_mex_clean(varargin)
cfg = gcmi_cpp_mex_config();
if exist(cfg.BuildDir, 'dir')
    rmdir(cfg.BuildDir, 's');
end
if exist(cfg.OutputDir, 'dir')
    rmdir(cfg.OutputDir, 's');
end
end
