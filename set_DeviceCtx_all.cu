#include <cuda.h>
#include <stdio.h>
#include <util.h>



int main (int argc, char * argv[]) {

    int n_dev;
    cudaGetDeviceCount(& n_dev);

    printf("%d devices are available\n", n_dev);
    if (n_dev <= 0) return 1;


    printf("Checking state of primary context -- before first CUDA runtime API call\n");
    check_primary_ctx(n_dev);


    printf("Checking state of primary context -- after a cudaMalloc on each device\n");

    CUcontext context[n_dev];
    CUdevice  device[n_dev];

    for (int i=0; i<n_dev; i++) {
        cuDeviceGet(& device[i], i);
        cuCtxCreate(& context[i], 0, device[i]);
        void * p;
        cudaMalloc(& p, 1);
    }

    check_primary_ctx(n_dev);

    return 0;
}
