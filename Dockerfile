# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

FROM ubuntu:22.04

# install dependencies 
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install --yes \
	build-essential \
    	direnv \
	git \
	m4 \
	scons \
	gettext \
	cmake \
	unzip \
	zlib1g \
	zlib1g-dev \
	libprotobuf-dev \
	protobuf-compiler \
	libprotoc-dev \
	libgoogle-perftools-dev \
	ninja-build \
	python3-dev \
	python3-pip \
	python3-venv \
	libboost-all-dev \
	pkg-config \
	npm \
	vim \
	wget \
	curl \
	zsh \
	gcc-9-aarch64-linux-gnu \
    	gfortran-aarch64-linux-gnu \
    	g++-aarch64-linux-gnu \
	tmux \
	ninja-build \
	gettext

# add some python dependencies 
COPY ./requirements.txt /
RUN pip install -r /requirements.txt && rm -f /requirements.txt

# (optional) configures zsh 
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
        -t af-magic \
        -p vi-mode \
        -p git \
        -a 'alias gem=/gem5/build/ARM/gem5.opt' 

# build gem5 
RUN mkdir /gem5 && git clone https://gem5.googlesource.com/public/gem5 /gem5
RUN cd /gem5 && git checkout 1db206b9d371b14cb9150e37aef444d1d59db025 && python3 `which scons` build/ARM/gem5.opt -j9

# build packer 
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim

# build neovim 
RUN git clone https://github.com/neovim/neovim /neovim && cd /neovim && git checkout stable && make CMAKE_BUILD_TYPE=RelWithDebInfo && make install

# get coremark 
RUN git clone https://github.com/eembc/coremark.git /coremark
