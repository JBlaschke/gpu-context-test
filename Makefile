NVCC     := nvcc
INCLUDE  := -I.

all: test_PrimaryCtx_0.ex test_PrimaryCtx_all.ex set_DeviceCtx_all.ex\
    set_DeviceCtx_0.ex get_DeviceCtx_0.ex test_SetDevice_all.ex\
    test_SetDevice_0.ex

%.ex: %.o util.o
	$(NVCC) -o $@ -lcuda $^

%.o: %.cu
	$(NVCC) -c $(INCLUDE) -o $@ $<
