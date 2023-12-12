FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY ./requirements.txt ./

RUN echo "Installing pip packages..." \
    && python -m pip install -U pip \
    && pip --no-cache-dir install -r requirements.txt \
    && rm ./requirements.txt

WORKDIR /workspace