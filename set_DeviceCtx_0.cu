#include <cuda.h>
#include <stdio.h>


typedef struct {
    int id;
    unsigned int flags;
    int active;
} primaryCtxState;



int main (int argc, char * argv[]) {

    int n_dev;
    cudaGetDeviceCount(& n_dev);

    printf("%d devices are available\n", n_dev);

    if (n_dev <= 0) return 1;


    primaryCtxState state[n_dev];


    printf("Checking state of primary context -- before first CUDA runtime API call\n");


    for (int i=0; i<n_dev; i++) {
        state[i].id = i;
        cuDevicePrimaryCtxGetState(
                state[i].id,
                & state[i].flags, & state[i].active
            );
    }

    for (int i=0; i<n_dev; i++) {
        printf(
                "Device %d state: flags=%d, active=%d\n",
                state[i].id, state[i].flags, state[i].active
            );
    }


    printf("Checking state of primary context -- after a cudaMalloc on each device\n");

    CUcontext context;
    CUdevice  device;

    cuDeviceGet(& device, 0);
    cuCtxCreate(& context, 0, device);
    void * p;
    cudaMalloc(& p, 1);

    for (int i=1; i<n_dev; i++) {
        cudaSetDevice(i);
        void * p;
        cudaMalloc(& p, 1);
    }

    cudaSetDevice(0);

    for (int i=0; i<n_dev; i++) {
        cuDevicePrimaryCtxGetState(
                state[i].id,
                & state[i].flags, & state[i].active
            );
    }

    for (int i=0; i<n_dev; i++) {
        printf(
                "Device %d state: flags=%d, active=%d\n",
                state[i].id, state[i].flags, state[i].active
            );
    }


    return 0;
}
