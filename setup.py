from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

NYUGYOKU_FEATURES = ('NYUGYOKU_FEATURES', None) if os.environ.get('NYUGYOKU_FEATURES') else None

class my_build_ext(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'unix':
            for e in self.extensions:
                e.extra_compile_args = ['-std=c++17', '-msse4.2', '-mavx2', '-fopenmp']
                e.extra_link_args = ['-lgomp']
        elif self.compiler.compiler_type == 'msvc':
            for e in self.extensions:
                e.extra_compile_args = ['/std:c++17', '/arch:AVX2', '/openmp']

        build_ext.build_extensions(self)

    def finalize_options(self):
        build_ext.finalize_options(self)
        try:
            __builtins__.__NUMPY_SETUP__ = False
        except:
            import builtins
            builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

ext_macros = [('HAVE_SSE4', None), ('HAVE_SSE42', None), ('HAVE_AVX2', None)]
if NYUGYOKU_FEATURES:
    ext_macros.append(NYUGYOKU_FEATURES)
ext_modules = [
    Extension('dlshogi.cppshogi',
        ['dlshogi/cppshogi.pyx',
         'cppshogi/cppshogi.cpp', 'cppshogi/python_module.cpp', 'cppshogi/bitboard.cpp', 'cppshogi/book.cpp', 'cppshogi/common.cpp', 'cppshogi/generateMoves.cpp', 'cppshogi/hand.cpp', 'cppshogi/init.cpp', 'cppshogi/move.cpp', 'cppshogi/mt64bit.cpp', 'cppshogi/position.cpp', 'cppshogi/search.cpp', 'cppshogi/square.cpp', 'cppshogi/usi.cpp'],
        language='c++',
        include_dirs = ["cppshogi"],
        define_macros=ext_macros)
]

setup(
    name = 'dlshogi',
    version = '0.1.1',
    author = 'Tadao Yamaoka',
    url='https://github.com/TadaoYamaoka/DeepLearningShogi',
    packages = ['dlshogi', 'dlshogi.network', 'dlshogi.utils'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': my_build_ext},
    description = 'DeepLearningShogi(dlshogi)',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License (GPL)',
        "Operating System :: OS Independent",
    ],
)
