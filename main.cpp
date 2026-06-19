#include <Python.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <climits>
#include <unistd.h>

bool call_run_pipeline(const std::string& a_path, const std::string& b_path,
                       const std::string& c_path,
                       int64_t M, int64_t N, int64_t K);

static std::string exe_dir(const char* argv0) {
    std::string p(argv0);
    auto pos = p.find_last_of('/');
    if (pos == std::string::npos) return ".";
    return p.substr(0, pos);
}

static std::string real_dir(const std::string& d) {
    char buf[PATH_MAX];
    if (realpath(d.c_str(), buf)) return buf;
    return d;
}

int main(int argc, char* argv[]) {
    std::string project_root = real_dir(exe_dir(argv[0]));

    // PYTHONPATH from environment is respected by Py_Initialize.
    // We just need to add the project root so that core.bridge is found.
    setenv("PYTHONPATH", project_root.c_str(), 0);

    Py_Initialize();

    std::cout << "[ENGINE] Project root: " << project_root << std::endl;

    std::string a_path = project_root + "/data/A.raw";
    std::string b_path = project_root + "/data/B.raw";
    std::string c_path = project_root + "/data/C.raw";
    int64_t M = 1024, N = 1024, K = 1024;

    if (argc >= 4) { a_path = argv[1]; b_path = argv[2]; c_path = argv[3]; }
    if (argc >= 7) { M = std::stoll(argv[4]); N = std::stoll(argv[5]); K = std::stoll(argv[6]); }

    std::cout << "[ENGINE] Pipeline: " << a_path << " @ " << b_path
              << " -> " << c_path << " (" << M << "x" << K << " @ " << K << "x" << N << ")" << std::endl;

    if (!call_run_pipeline(a_path, b_path, c_path, M, N, K)) {
        std::cerr << "[ENGINE] ERROR: Pipeline fallo." << std::endl;
        Py_Finalize();
        return 1;
    }

    Py_Finalize();
    std::cout << "[ENGINE] Pipeline completado exitosamente." << std::endl;
    return 0;
}
