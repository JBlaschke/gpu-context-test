#include <util.h>
#include <cuda.h>
#include <stdio.h>



void check_primary_ctx(int n_dev) {

    primaryCtxState state[n_dev];

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
}


void api_malloc(int ** d_p, int M) {
    cudaMalloc(d_p, sizeof(int)*M);
}



void api_HtoD(int * h_p, int * d_p, int M) {
    cudaMemcpy(d_p, h_p, sizeof(int)*M, cudaMemcpyHostToDevice);
}



void api_DtoH(int * h_p, int * d_p, int M) {
    cudaMemcpy(h_p, d_p, sizeof(int)*M, cudaMemcpyDeviceToHost);
}



__global__
void saxpy_int(int n, int a, int * x, int * y) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}
