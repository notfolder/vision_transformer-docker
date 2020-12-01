FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 0 && \
    update-alternatives --set python /usr/bin/python3 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 && \
    update-alternatives --set pip /usr/bin/pip3 && \
    pip install --upgrade pip

RUN pip install numpy==1.18.5 scipy six==1.15.0 wheel

COPY jaxlib-*.whl /tmp/
RUN pip install /tmp/jaxlib-*.whl

RUN apt-get install -y wget git
WORKDIR /root
RUN git clone https://github.com/google-research/vision_transformer.git
WORKDIR /root/vision_transformer
RUN pip install -r vit_jax/requirements.txt

COPY entrypoint.py /root/vision_transformer/
RUN chmod +x /root/vision_transformer/entrypoint.py
ENTRYPOINT ["/root/vision_transformer/entrypoint.py"]
