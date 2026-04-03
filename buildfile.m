function plan = buildfile
plan = buildplan(localfunctions);
plan.DefaultTasks = "test";
plan("test").Dependencies = "compile";
plan("bench").Dependencies = "compile";
plan("package").Dependencies = "compile";
end

function compileTask(~)
add_tooling_path();
gcmi_cpp_mex_compile();
end

function testTask(~)
add_tooling_path();
gcmi_cpp_mex_test();
end

function benchTask(~)
root = fileparts(mfilename("fullpath"));
addpath(fullfile(root, "matlab", "cpp_mex", "tooling"));
addpath(fullfile(root, "benchmarks"));
fmap = cpp_mex_default_function_map();
run_matlab_benchmarks( ...
    "FixtureIds", {"cc_small_f64", "cd_medium_f64"}, ...
    "OptimizedLabel", "cpp_mex", ...
    "OptimizedFunctions", fmap, ...
    "OptimizedPaths", fullfile(root, "matlab", "cpp_mex", "bin", version("-release"), mexext), ...
    "OptimizedLabelEncoding", "zero_based");
end

function packageTask(~)
add_tooling_path();
gcmi_cpp_mex_package();
end

function cleanTask(~)
add_tooling_path();
gcmi_cpp_mex_clean();
end

function add_tooling_path()
root = fileparts(mfilename("fullpath"));
addpath(fullfile(root, "matlab", "cpp_mex", "tooling"));
end
