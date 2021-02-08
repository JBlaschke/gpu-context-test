#ifndef __UTIL_H__
#define __UTIL_H__


typedef struct primaryCtxState {
    int id;
    unsigned int flags;
    int active;
} primaryCtxState;

void check_primary_ctx(int n_dev);

void api_malloc(int ** d_p, int M);
void api_HtoD(int * h_p, int * d_p, int M);
void api_DtoH(int * h_p, int * d_p, int M);

__global__
void saxpy_int(int n, int a, int * x, int * y);

#endif
