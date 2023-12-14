## HaUI human sperm classification

### Author
* Vũ Việt Thắng, HaUI, (vuvietthang@haui.edu.vn) 
* Đỗ Mạnh Quang, HaUI, (domanhquang.haui@gmail.com)

### Install Lib
```shell
git clone https://github.com/DoManhQuang/human-sperm-classification.git
cd human-sperm-classification
pip install -r requirement.txt
```

### Train
```shell
python tools/train.py \
-v "version-0.0" -ep 10 -bsize 16 -verbose 1 \
-train "./dataset/smids/smids_datatrain.data" \
-test "./dataset/smids/smids_datatest.data" \
-activation_block "relu" \
--mode_model "model-base" \
--result_path "./runs/results" \
--training_path "./runs/training" \
-name "smids-dataset"
```

### K-Fold
```shell
python tools/kfold.py \
-v "version-0.1" \
-ep 5 -bsize 16 -verbose 1 \
-train "dataset/smids_datatrain.data" \
-test "dataset/smids_datatest.data" \
-name "name-kfold" \
-nk 10 -ck 1 \
--activation_block "relu" \
--mode_model "model-base"
```
