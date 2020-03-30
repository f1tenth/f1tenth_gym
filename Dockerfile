FROM ubuntu:18.04

ENV IM_IN_DOCKER Yes

RUN apt-get update --fix-missing && \
    apt-get install -y \
    python3-dev python3-pip

RUN apt-get install -y libzmq3-dev \
                       nano \
                       git \
                       unzip \
                       build-essential \
                       autoconf \
                       libtool \
                       libeigen3-dev \
                       cmake \
                       emacs

RUN cp -r /usr/include/eigen3/Eigen /usr/include

RUN git clone https://github.com/google/protobuf.git && \
    cd protobuf && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    make clean && \
    cd .. && \
    rm -r protobuf

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                scipy \
                numba \
                matplotlib \
                zmq \
                pyzmq \
                Pillow \
                gym \
                protobuf \
                pyyaml \
                msgpack==0.6.2

RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /simulator
COPY . /simulator

RUN cd /simulator && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make

RUN cp /simulator/build/sim_requests_pb2.py /simulator/gym/

RUN cd /simulator && \
    pip3 install -e gym/

WORKDIR /simulator

EXPOSE 5557
EXPOSE 5558


ENTRYPOINT ["/bin/bash"]