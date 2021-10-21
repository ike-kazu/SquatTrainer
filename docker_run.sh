#! /bin/bash

sudo nvpmodel -m 0
sudo jetson_clocks

docker run -it \
	--runtime nvidia \
	--network host \
	--rm \
	-v $PWD/app:/usr/app \
	-v $PWD/web/nginx.conf:/etc/nginx/nginx.conf \
	-v $PWD/web/web.conf:/etc/nginx/conf.d/web.conf \
	-v $PWD/log:/var/log/nginx \
	--device /dev/video0 \
	--name my_app tlapesium/traintraing:latest

