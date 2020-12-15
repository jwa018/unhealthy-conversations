FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
WORKDIR /tmp
ADD requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /