FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install git sudo -y && \
  pip3 install --upgrade pip && \
  mkdir /usr/app && \
  cd && \
  git config --global http.sslVerify false && \
  git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
  cd torch2trt && \
  python3 setup.py install --plugins && \
  cd && \
  pip3 install tqdm cython pycocotools && \
  apt-get install python3-matplotlib python3-tk tk-dev -y && \
  git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
  cd trt_pose && \
  python3 setup.py install && \
  git clone https://github.com/NVIDIA-AI-IOT/jetcam && \
  cd jetcam && \
  sudo python3 setup.py install && \
  cd && \
  pip3 install ipywidgets && \
  jupyter nbextension enable --py widgetsnbextension && \
  wget https://nodejs.org/dist/v12.13.0/node-v12.13.0-linux-arm64.tar.xz && \
  tar -xJf node-v12.13.0-linux-arm64.tar.xz && \
  cd node-v12.13.0-linux-arm64 && \
  cp -R * /usr/local/ && \
  cd && \
  jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install nginx -y && \
  pip3 install -U Pygments && \
  pip3 install uwsgi && \
  pip3 install flask && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

CMD ["bash"]
