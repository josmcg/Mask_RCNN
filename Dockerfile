FROM nvidia/cuda:10.0-baseubuntu16.04
RUN mkdir /conda
WORKDIR /conda
COPY requirements.txt .
RUN curl -so miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x miniconda.sh \
 && miniconda.sh -b -p /conda/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/conda/miniconda/bin:$PATH
RUN conda create -yn tflow python=3.6
RUN source activate tflow; pip install -r requirements.txt
mkdir /vol


