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
