KERNEL_NAME := edge_conv_compute_kernel
PLATFORM := /opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm

KERNEL_SRC := \
	src/edge_conv_kernel.cpp \
	src/edge_conv.cpp \
	src/finalize.cpp \
	src/globals.cpp \
	src/load_inputs.cpp \
	src/message_passing.cpp \
	src/mp_to_ne_adapter.cpp \
	src/node_transformation.cpp \
	src/operators.cpp

HOST_SRC := \
	src/host.cpp \
	src/host_load.cpp \
	src/host_helper.cpp \
	src/xcl2/xcl2.cpp

XO := $(KERNEL_NAME).xo
XCLBIN := $(KERNEL_NAME).xclbin
CONFIG_LINK := config_link.cfg
HOST_EXE := host

PROFILING_FLAGS := --profile.data all:all:all --profile.exec all:all --profile.stall all:all
opencl_CXXFLAGS := -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -I$(XILINX_VITIS)/include -L$(XILINX_VITIS)/lib -I$(XILINX_VIVADO)/include
opencl_LDFLAGS := -L$(XILINX_XRT)/lib -lOpenCL -lxrt_coreutil -lpthread
CXXFLAGS := $(opencl_CXXFLAGS) -Wall -O0 -g -Isrc/xcl2 -Isrc/opencl -Isrc/oclHelper -fmessage-length=0 -Wno-int-in-bool-context -Wno-unused-result -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS := $(opencl_LDFLAGS) -Wl,--no-as-needed -lrt -lstdc++

TARGET := hw
FREQ_HZ := 200000000
FREQ_MHZ := 200


.PHONY: compile_complete link_complete host_compile host_emu host_emu_complete host_run host_complete emu_all all

compile_complete:
	v++ --target $(TARGET) --compile --platform $(PLATFORM) --kernel $(KERNEL_NAME) --output $(XO) $(KERNEL_SRC) --include ./src/ --hls.clock $(FREQ_HZ):$(KERNEL_NAME)


link_complete:
	v++ -g --target $(TARGET) --link --platform $(PLATFORM) $(XO) --output $(XCLBIN) --kernel_frequency $(FREQ_MHZ) --clock.defaultFreqHz $(FREQ_HZ) --config $(CONFIG_LINK) $(PROFILING_FLAGS)


host_compile:
	g++ -g -o $(HOST_EXE) $(HOST_SRC) $(CXXFLAGS) $(LDFLAGS)

host_emu:
	XCL_EMULATION_MODE=sw_emu ./$(HOST_EXE) $(XCLBIN)
host_emu_complete: host_compile host_emu

host_run:
	./$(HOST_EXE) $(XCLBIN)
host_complete: host_compile host_run


# make emu_all TARGET=sw_emu
emu_all: compile_complete link_complete host_emu_complete

all: compile_complete link_complete host_complete