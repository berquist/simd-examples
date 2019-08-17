project('simd-examples', 'cpp',
        default_options : ['cpp_std=c++17'])
benchmark = dependency('benchmark')
executable('simd_examples.x', 'src/main.cpp', dependencies : benchmark, cpp_args : '-mavx2')
