#ifndef __UTIL_H__
#define __UTIL_H__


typedef struct primaryCtxState {
    int id;
    unsigned int flags;
    int active;
} primaryCtxState;

void check_primary_ctx(int n_dev);

#endif
