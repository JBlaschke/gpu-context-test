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



CUresult get_current_device(int * i_dev, int * is_primary, int * is_clean) {

    cudaError_t cuda_err;
    CUresult ierr;

    // GET current device bound to this thread (this will work for _both_ the
    // cuda runtim API and the cuda driver)
    cuda_err = cudaGetDevice(i_dev);
    if (cuda_err != cudaSuccess) {
        * i_dev      = -1;
        * is_primary = -1;
        * is_clean   = -1;
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    // GET the state of the primary context
    unsigned int flags;
    int active;
    cuDevicePrimaryCtxGetState(* i_dev, & flags, & active);
    if (active == 1){
        * is_primary = 1;
        * is_clean   = 0;
        return CUDA_SUCCESS;
    }

    // The PRIMARY CONTEXT could be the only context on the device bound to
    // this thread -- and just not be active because nothing has been called
    // it, or another context is the current context

    CUdevice device;

    ierr = cuCtxGetDevice(& device);
    if (ierr == CUDA_ERROR_INVALID_CONTEXT) {
        * is_primary = 1;
        * is_clean   = 1;
    } else if (ierr != CUDA_SUCCESS) {
        return ierr;
    }

    // There is defintely a device bound to this thread -- so there MUST be a
    // current context. The only thing to figure out now is if this context is
    // the primary context (which just hasn't been used yet), or another.

    CUcontext context;

    ierr = cuCtxGetCurrent(& context);
    if (ierr != CUDA_SUCCESS)
        return ierr;

    CUcontext primary_context;
    // NOTE: this will make the primary context active -- we need to release it
    // again below:
    ierr = cuDevicePrimaryCtxRetain(& primary_context, device);
    if (ierr != CUDA_SUCCESS)
        return ierr;
    // Restore the pre-existing context by de-activating the primary context
    // that cudaPrimaryCtxRetain activated
    ierr = cuDevicePrimaryCtxRelease(device);
    if (ierr != CUDA_SUCCESS)
        return ierr;

    if (primary_context == context) {
        * is_primary = 1;
        * is_clean   = 0;

        return CUDA_SUCCESS;
    }

    * is_primary = 0;
    * is_clean   = 0;

    return CUDA_SUCCESS;
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
