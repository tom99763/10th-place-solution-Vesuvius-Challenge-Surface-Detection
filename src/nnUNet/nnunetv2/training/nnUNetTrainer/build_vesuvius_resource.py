import subprocess
import sys
import importlib
import os
import platform


def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def install_dependencies():
    """Installs topometrics library. This version works on Windows by replacing
    chmod + make + shell scripts with a CMake/MSVC build pipeline."""
    try:
        import topometrics.leaderboard
        return None
    except Exception:
        pass

    resources_dir = './vesuvius_resource'
    install_dir = './vesuvius_resource'

    try:
        # ---------------------------
        # 1. Build Betti (Windows version)
        # ---------------------------
        if platform.system() == "Windows":
            build_dir = os.path.join(install_dir, "build")

            # Create build directory
            if not os.path.exists(build_dir):
                os.makedirs(build_dir)

            print(">>> Configuring CMake build (Windows)...")
            run("cmake -S . -B build", cwd=install_dir)

            print(">>> Building Betti with MSVC...")
            run("cmake --build build --config Release", cwd=install_dir)

        else:
            # Linux/Mac branch (original)
            run(
                "chmod +x scripts/setup_submodules.sh scripts/build_betti.sh && make build-betti",
                cwd=install_dir
            )

        # ---------------------------
        # 2. Install topometrics (editable, no deps)
        # ---------------------------
        print(">>> Installing topometrics...")
        run(
            "uv pip install -e . --no-deps --no-index --no-build-isolation -v",
            cwd=install_dir
        )

        # ---------------------------
        # 3. Add path so Python can find it
        # ---------------------------
        sys.path.append(f'./{resources_dir}/topological-metrics-kaggle/src')
        importlib.invalidate_caches()

    except Exception as err:
        raise Exception(f'Failed to install topometrics library: {err}')


if __name__ == '__main__':
    install_dependencies()