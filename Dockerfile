FROM nvidia/cuda:10.0-base-ubuntu16.04
RUN mkdir /conda
WORKDIR /conda
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y wget bzip2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
RUN chmod +x /conda/miniconda.sh 
RUN /conda/miniconda.sh -b -p /conda/miniconda 
RUN rm /conda/miniconda.sh
ENV PATH=/conda/miniconda/bin:$PATH
RUN conda create -yn tflow python=3.6
RUN bash -c "source activate tflow; pip install -r requirements.txt"
RUN bash -c "source activate tflow; conda install tensorflow-gpu==1.8.0"
RUN mkdir  /vol
WORKDIR /vol


