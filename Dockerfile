FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /usr/src/app

ENV DISTRO ubuntu1804
ENV CPU_ARCH x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$CPU_ARCH/3bf863cc.pub

RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt ./
RUN pip3 install --upgrade "pip < 21.0"
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY . .

CMD [ "python3", "./server.py" ]

