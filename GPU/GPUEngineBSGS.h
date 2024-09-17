#ifndef GPU_ENGINE_BSGS_H
#define GPU_ENGINE_BSGS_H

#include <cstdint>

class GPUEngineBSGS
{
public:
    GPUEngineBSGS(int device_id, int m);
    ~GPUEngineBSGS();

    void setParameters(uint64_t *G, uint64_t *mG);
    void generateBabySteps();
    void sortBabySteps();
    bool solve(uint64_t *target, uint64_t *result);

private:
    int m; // Size of baby steps

    uint64_t *d_babySteps;
    uint64_t *d_G;
    uint64_t *d_mG;
    uint64_t *d_target;
    uint64_t *d_result;
    int *d_found;
};

#endif // GPU_ENGINE_BSGS_H