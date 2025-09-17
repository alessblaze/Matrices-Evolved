from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import nanobind
import subprocess
import os
import shutil

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        source_dir = os.path.abspath('.')
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)
        
        # Use system cmake directly
        cmake_cmd = '/usr/bin/cmake'
        
        subprocess.run([cmake_cmd, source_dir, '-DCMAKE_BUILD_TYPE=Release'], 
                      cwd=build_dir, check=True)
        subprocess.run([cmake_cmd, '--build', '.'], 
                      cwd=build_dir, check=True)
        
        # Copy the built libraries to the right location
        dest_dir = os.path.join(self.build_lib, 'matrices_evolved')
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy C++ module
        built_lib = os.path.join(build_dir, '_event_signing_impl*.so')
        import glob
        for lib_file in glob.glob(built_lib):
            dest = os.path.join(dest_dir, os.path.basename(lib_file))
            shutil.copy2(lib_file, dest)
        
        # Copy Rust module if it exists
        rust_lib = os.path.join(build_dir, 'matrices_evolved_rust.so')
        if os.path.exists(rust_lib):
            # Rename to Python-compatible module name
            dest = os.path.join(dest_dir, 'matrices_evolved_rust.cpython-310-x86_64-linux-gnu.so')
            shutil.copy2(rust_lib, dest)
            print(f"Copied Rust module: {rust_lib} -> {dest}")
        else:
            print(f"Rust module not found at: {rust_lib}")

setup(
    name="matrices_evolved",
    version="1.0.0",
    ext_modules=[Extension("matrices_evolved._event_signing_impl", [])],
    cmdclass={"build_ext": CMakeBuild},
    packages=["matrices_evolved"],
    install_requires=["nanobind>=2.0.0"],
    zip_safe=False,
)