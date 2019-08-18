project('simd-examples', 'cpp',
        default_options : ['cpp_std=c++17'])
benchmark = dependency('benchmark')
executable('simd_examples.x', 'src/functions.cpp', 'src/main.cpp',
           dependencies : benchmark,
           cpp_args : ['-mavx2'])
           # cpp_args : ['-mavx2', '-O1'])
