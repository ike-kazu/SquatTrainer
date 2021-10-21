## 起動方法
ホストで以下のコマンドを実行
```
$ docker pull tlapesium/traintraing:latest
$ sudo bash docker_run.sh
```
コンテナで以下のコマンドを実行
```
$ bash run.sh
```

## ビルド方法
dockerのデフォルトランタイムをnvidiaにしておく。
```
$ cd build
$ sudo bash docker_build.sh
```

## その他
ホームディレクトリにtrt\_poseなどが置いてあるので、trt\_poseのサンプルを実行できます。
コンテナで以下のコマンドを実行し、jupyter labで `live_demo.ipynb` を実行するとよいです。
```
$ cp ./resnet18_baseline_att_224x224_A_epoch_249.pth ~/trt_pose/tasks/human_pose/
$ cd ~/trt_pose/tasks/human_pose/
$ jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

