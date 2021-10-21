#! /bin/bash

uwsgi --ini myapp.ini &
/etc/init.d/nginx start