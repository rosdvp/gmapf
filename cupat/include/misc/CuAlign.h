#pragma once

#if defined(__CUDACC__) // NVCC
#define CU_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define CU_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define CU_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif