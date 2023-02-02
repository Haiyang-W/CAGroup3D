from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KNN_OP',
    ext_modules=[
        CUDAExtension('KNN_OP', [
            'src/knn.cpp',
            'src/knn_cuda.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)


# from setuptools import find_packages, setup

# import os
# import torch
# from torch.utils.cpp_extension import (BuildExtension, CppExtension,
#                                        CUDAExtension)

# def make_cuda_ext(name,
#                   module,
#                   sources,
#                   sources_cuda=[],
#                   extra_args=[],
#                   extra_include_path=[]):

#     define_macros = []
#     extra_compile_args = {'cxx': [] + extra_args}

#     if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
#         define_macros += [('WITH_CUDA', None)]
#         extension = CUDAExtension
#         extra_compile_args['nvcc'] = extra_args + [
#             '-D__CUDA_NO_HALF_OPERATORS__',
#             '-D__CUDA_NO_HALF_CONVERSIONS__',
#             '-D__CUDA_NO_HALF2_OPERATORS__',
#         ]
#         sources += sources_cuda
#     else:
#         print('Compiling {} without CUDA'.format(name))
#         extension = CppExtension
#         # raise EnvironmentError('CUDA is required to compile MMDetection!')

#     return extension(
#         name='{}.{}'.format(module, name),
#         sources=[os.path.join(*module.split('.'), p) for p in sources],
#         include_dirs=extra_include_path,
#         define_macros=define_macros,
#         extra_compile_args=extra_compile_args)

# setup(
#     name='knn_ext',
#     ext_modules=[
#         make_cuda_ext(
#                 name='knn_ext',
#                 module='pcdet.ops.knn',
#                 sources=['src/knn.cpp'],
#                 sources_cuda=['src/knn_cuda.cu']),
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )

