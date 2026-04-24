from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, torch

torch_lib = os.path.join(torch.__path__[0], 'lib')  # …/site-packages/torch/lib

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1'],
    'nvcc': [
        '-O3', '--use_fast_math', '-std=c++17',
        '-gencode=arch=compute_80,code=sm_80',
        '-Xcompiler', '-fPIC',
        '--expt-relaxed-constexpr',
        '-lineinfo',
    ],
}
# 直接把 torch/lib 寫進 rpath，避免 ImportError: libc10.so
extra_link_args = [f'-Wl,-rpath,{torch_lib}', '-Wl,-rpath,$ORIGIN']

setup(
    name='elsa_ext',
    version='0.3.0',
    packages=[],      # 避免掃到你 repo 其他頂層包（timm/transformers…）
    py_modules=[],
    ext_modules=[
        CUDAExtension(
            name='elsa_ext',
            sources=['../elsa.cpp', '../elsa_kernel.cu'],  # 依你檔案實際位置調整
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
