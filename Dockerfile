FROM python:3.11-slim

RUN pip install --upgrade pip

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential git-core \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ADD . /root/ytmate
#Important: secrets and requirements_private.txt need to be in the same folder
ADD deploy/files/ deploy/.gitconfig* /root/


RUN python -m pip install -r /root/requirements.txt --no-cache-dir

WORKDIR /root/ytmate/

EXPOSE 3000
