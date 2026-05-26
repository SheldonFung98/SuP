import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/DNOMINMAX", "/EHsc"],
        "nvcc": ["-O3", "-std=c++17"],
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"],
    }


setup(
    name='geotransformer',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extensions/extra/cloud/cloud.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_dps.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu_dps.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/extensions/pybind.cpp',
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
