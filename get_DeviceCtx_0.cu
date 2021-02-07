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

    int cu_dev = 1;

    CUcontext context;
    CUdevice  device;

    cuDeviceGet(& device, cu_dev);
    cuCtxCreate(& context, 0, device);

    int ierr = cuCtxGetDevice(& device);
    printf("%d, %d\n", device, ierr);

    int dev;
    cudaGetDevice(& dev);
    printf("cuda api dev=%d\n", dev);

    for (int i=0; i<n_dev; i++) {
        if (i != cu_dev) cudaSetDevice(i);
        int * d_p;
        cudaMalloc(& d_p, sizeof(int));
        int payload = 10;
        int * p_payload = & payload;
        cudaMemcpy(d_p, p_payload, sizeof(int), cudaMemcpyHostToDevice);
        int r_payload = 0;
        int * p_r_payload = & r_payload;
        cudaMemcpy(p_r_payload, d_p, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Device %d round-trip int=%d\n", i, r_payload);
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
