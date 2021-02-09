#ifndef __UTIL_H__
#define __UTIL_H__

#include <cuda.h>


typedef struct primaryCtxState {
    int id;
    unsigned int flags;
    int active;
} primaryCtxState;

void check_primary_ctx(int n_dev);
CUresult get_current_device(int * i_dev, int * is_primary, int * is_clean);

void api_malloc(int ** d_p, int M);
void api_HtoD(int * h_p, int * d_p, int M);
void api_DtoH(int * h_p, int * d_p, int M);

__global__
void saxpy_int(int n, int a, int * x, int * y);

#endif
