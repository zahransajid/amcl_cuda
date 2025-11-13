#pragma once
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(call)                                                                       \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = call;                                                                    \
        if (cudaSuccess != err)                                                                    \
        {                                                                                          \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                                                       \
    do                                                                                             \
    {                                                                                              \
        /* Check synchronous errors, i.e. pre-launch */                                            \
        cudaError_t err = cudaGetLastError();                                                      \
        if (cudaSuccess != err)                                                                    \
        {                                                                                          \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        /* Check asynchronous errors, i.e. kernel failed (ULF) */                                  \
        err = cudaDeviceSynchronize();                                                             \
        if (cudaSuccess != err)                                                                    \
        {                                                                                          \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
