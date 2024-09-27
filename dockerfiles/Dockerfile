FROM ubuntu:20.04 as vina-gpu

ARG DEBIAN_FRONTEND=noninteractive
ARG WORKDIR="/vina-gpu-dockerized"
ARG BOOST_DIR_NAME="boost_1_77_0"


RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt \
    apt-get update && apt-get -y upgrade && apt-get install -y \
    clinfo \
    cmake \
    ocl-icd-libopencl1 \
    opencl-headers \
    python3-pip \
    tar \
    wget \
    xz-utils

RUN mkdir -p /etc/OpenCL/vendors && \
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR "${WORKDIR}"
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/${BOOST_DIR_NAME}.tar.bz2 && tar --bzip2 -xf ${BOOST_DIR_NAME}.tar.bz2 && rm ${BOOST_DIR_NAME}.tar.bz2

WORKDIR "${WORKDIR}/${BOOST_DIR_NAME}"
RUN mkdir ${WORKDIR}/${BOOST_DIR_NAME}/build
RUN cd tools/build && ./bootstrap.sh && ./b2 install --prefix=${WORKDIR}/${BOOST_DIR_NAME}/build
# TODO: most likely here we can select only some parts of the library to be built
RUN ${WORKDIR}/${BOOST_DIR_NAME}/build/bin/b2 --build-dir=${WORKDIR}/${BOOST_DIR_NAME}/build toolset=gcc stage

RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WORKDIR}/${BOOST_DIR_NAME}/stage/lib"
COPY . ${WORKDIR}/vina
WORKDIR "${WORKDIR}/vina"

RUN gcc -o Vina-GPU \
        -I${WORKDIR}/${BOOST_DIR_NAME} -I./lib -I./OpenCL/inc \
        ./main/main.cpp \
        -O3 ./lib/*.cpp ./OpenCL/src/wrapcl.cpp ${WORKDIR}/${BOOST_DIR_NAME}/libs/thread/src/pthread/thread.cpp ${WORKDIR}/${BOOST_DIR_NAME}/libs/thread/src/pthread/once.cpp \
        -lboost_program_options -lboost_system -lboost_filesystem -lOpenCL -lstdc++ -lm -lpthread \
        -L${WORKDIR}/${BOOST_DIR_NAME}/stage/lib -L/usr/lib/x86_64-linux-gnu \
        -DOPENCL_1_2 -DBUILD_KERNEL_FROM_SOURCE -DNVIDIA_PLATFORM

