#include "GPUEngineBSGS.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"
#include "HashTable.h"

#include "GPUBase58.h"

#define BLOCK_SIZE 256

// CUDA kernel for generating baby steps
__global__ void generateBabyStepsKernel(uint64_t *babySteps, int m, uint64_t *G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        uint64_t point[4];
        Load256(point, G);

        for (int i = 0; i < idx; i++)
        {
            _ModMult(point, G); // Use existing ModMult operation
        }

        // Store the resulting point
        Store256A(babySteps + idx * 4, point);
    }
}

// CUDA kernel for sorting baby steps (using bitonic sort for better performance)
__global__ void bitonicSortKernel(uint64_t *babySteps, int j, int k)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            if (babySteps[i * 4] > babySteps[ixj * 4])
            {
                for (int t = 0; t < 4; t++)
                {
                    uint64_t temp = babySteps[i * 4 + t];
                    babySteps[i * 4 + t] = babySteps[ixj * 4 + t];
                    babySteps[ixj * 4 + t] = temp;
                }
            }
        }
        else
        {
            if (babySteps[i * 4] < babySteps[ixj * 4])
            {
                for (int t = 0; t < 4; t++)
                {
                    uint64_t temp = babySteps[i * 4 + t];
                    babySteps[i * 4 + t] = babySteps[ixj * 4 + t];
                    babySteps[ixj * 4 + t] = temp;
                }
            }
        }
    }
}

// CUDA kernel for giant step computation and matching
__global__ void giantStepKernel(uint64_t *babySteps, int m, uint64_t *target, uint64_t *mG, uint64_t *result, int *found)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        uint64_t point[4];
        Load256(point, target);

        for (int i = 0; i < idx; i++)
        {
            _ModSub(point, mG); // Use existing ModSub operation
        }

        // Binary search in baby steps
        int low = 0, high = m - 1;
        while (low <= high)
        {
            int mid = (low + high) / 2;
            if (_IsEqual(babySteps + mid * 4, point))
            {
                *found = 1;
                result[0] = idx;
                result[1] = mid;
                return;
            }
            if (_IsLower(babySteps + mid * 4, point))
                low = mid + 1;
            else
                high = mid - 1;
        }
    }
}

GPUEngineBSGS::GPUEngineBSGS(int device_id, int m) : m(m)
{
    cudaSetDevice(device_id);
    cudaMalloc(&d_babySteps, m * 4 * sizeof(uint64_t));
    cudaMalloc(&d_G, 4 * sizeof(uint64_t));
    cudaMalloc(&d_mG, 4 * sizeof(uint64_t));
    cudaMalloc(&d_target, 4 * sizeof(uint64_t));
    cudaMalloc(&d_result, 2 * sizeof(uint64_t));
    cudaMalloc(&d_found, sizeof(int));
}

GPUEngineBSGS::~GPUEngineBSGS()
{
    cudaFree(d_babySteps);
    cudaFree(d_G);
    cudaFree(d_mG);
    cudaFree(d_target);
    cudaFree(d_result);
    cudaFree(d_found);
}

void GPUEngineBSGS::setParameters(uint64_t *G, uint64_t *mG)
{
    cudaMemcpy(d_G, G, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mG, mG, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

void GPUEngineBSGS::generateBabySteps()
{
    int gridSize = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generateBabyStepsKernel<<<gridSize, BLOCK_SIZE>>>(d_babySteps, m, d_G);
    cudaDeviceSynchronize();
}

void GPUEngineBSGS::sortBabySteps()
{
    for (int k = 2; k <= m; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            int gridSize = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bitonicSortKernel<<<gridSize, BLOCK_SIZE>>>(d_babySteps, j, k);
            cudaDeviceSynchronize();
        }
    }
}

bool GPUEngineBSGS::solve(uint64_t *target, uint64_t *result)
{
    cudaMemcpy(d_target, target, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int gridSize = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int found = 0;
    cudaMemcpy(d_found, &found, sizeof(int), cudaMemcpyHostToDevice);

    giantStepKernel<<<gridSize, BLOCK_SIZE>>>(d_babySteps, m, d_target, d_mG, d_result, d_found);
    cudaDeviceSynchronize();

    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (found)
    {
        cudaMemcpy(result, d_result, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return true;
    }
    return false;
}