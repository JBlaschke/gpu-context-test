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


    printf("Checking state of primary context -- after saxpy on each device sequentially\n");

    int M = 1000;
    int * h_x = (int *) malloc(M*sizeof(int));
    int * h_y = (int *) calloc(M, sizeof(int));
    for (int i=0; i<M; i++) h_x[i] = 10;

    CUcontext context;
    CUdevice  device;

    cuDeviceGet(& device, 0);
    cuCtxCreate(& context, 0, device);
    cuCtxPushCurrent(context);

    printf("NOTE: NOT calling cudaSetDevice on device 0\n");
    for (int i=0; i<n_dev; i++) {
        if (i>0) cudaSetDevice(i);

        int * d_x, * d_y;
        api_malloc(& d_x, M);
        api_malloc(& d_y, M);
        api_HtoD(h_x, d_x, M);
        api_HtoD(h_y, d_y, M);
        saxpy_int<<<(M+255)/256, 256>>>(M, 1, d_x, d_y);
        api_DtoH(h_x, d_x, M);
        api_DtoH(h_y, d_y, M);
        cudaFree(d_x);
        cudaFree(d_y);
        printf("Device %d work result: %d\n", i, h_y[0]);
    }

    check_primary_ctx(n_dev);

    printf("NOW calling cudaSetDevice on device 0\n");
    cudaSetDevice(0);
    int * d_x, * d_y;
    api_malloc(& d_x, M);
    api_malloc(& d_y, M);
    api_HtoD(h_x, d_x, M);
    api_HtoD(h_y, d_y, M);
    saxpy_int<<<(M+255)/256, 256>>>(M, 1, d_x, d_y);
    api_DtoH(h_x, d_x, M);
    api_DtoH(h_y, d_y, M);
    cudaFree(d_x);
    cudaFree(d_y);
    printf("Device %d work result: %d\n", 0, h_y[0]);
    check_primary_ctx(n_dev);

    free(h_x);
    free(h_y);
    return 0;
}
