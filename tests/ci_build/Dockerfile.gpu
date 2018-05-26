FROM nvidia/cuda:8.0-cudnn7-devel

# Base scripts
RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

COPY install/ubuntu_install_llvm.sh /install/ubuntu_install_llvm.sh
RUN bash /install/ubuntu_install_llvm.sh

COPY install/ubuntu_install_opencl.sh /install/ubuntu_install_opencl.sh
RUN bash /install/ubuntu_install_opencl.sh

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

COPY install/ubuntu_install_sphinx.sh /install/ubuntu_install_sphinx.sh
RUN bash /install/ubuntu_install_sphinx.sh

# Fix recommonmark to latest version
RUN git clone https://github.com/rtfd/recommonmark
RUN cd recommonmark; python setup.py install

# Enable doxygen for c++ doc build
RUN apt-get update && apt-get install -y doxygen graphviz libprotobuf-dev protobuf-compiler

# DL Frameworks
COPY install/ubuntu_install_mxnet.sh /install/ubuntu_install_mxnet.sh
RUN bash /install/ubuntu_install_mxnet.sh

COPY install/ubuntu_install_onnx.sh /install/ubuntu_install_onnx.sh
RUN bash /install/ubuntu_install_onnx.sh

COPY install/ubuntu_install_coreml.sh /install/ubuntu_install_coreml.sh
RUN bash /install/ubuntu_install_coreml.sh

COPY install/ubuntu_install_keras.sh /install/ubuntu_install_keras.sh
RUN bash /install/ubuntu_install_keras.sh

COPY install/ubuntu_install_darknet.sh /install/ubuntu_install_darknet.sh
RUN bash /install/ubuntu_install_darknet.sh

RUN pip install Pillow

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
