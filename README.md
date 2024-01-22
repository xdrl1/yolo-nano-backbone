# Modified YOLOX-Nano for 2D object detection on COCO

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Acknowledgement](#acknowledgement)

### Steps

1. docker

    ```bash
    docker run --gpus device=0 -d -it --shm-size 32G --user root --mount source=$(pwd),target=/home/vitis-ai-user/yolonano,type=bind tumbgd/vai-pt-cuda
    docker exec -it <docker-container-id> bash
    ```
clone to repo to the current path (/home/vitis-ai-user/yolonano).

2. within the docker container, install dependencies
    ```bash
    cd ~/yolonano/yolo-nano-backbone
    pip install --user -r requirements.txt
    sudo chmod -R 777 code
    cd code
    pip install --user -v -e .
    cd ..
    ```

### Preparation

1. Dataset description

The dataset MSCOCO2017 contains 118287 images for training and 5000 images for validation.

2. Download COCO dataset and create directories like this:
  ```plain
  └── data
       └── COCO
             ├── annotations
             |   ├── instances_train2017.json
             |   ├── instances_val2017.json
             |   └── ...
             ├── train2017
             |   ├── 000000000009.jpg
             |   ├── 000000000025.jpg
             |   ├── ...
             ├── val2017
                 ├── 000000000139.jpg
                 ├── 000000000285.jpg
             |   ├── ...
             └── test2017
                 ├── 000000000001.jpg
                 ├── 000000000016.jpg
                 └── ...
  ```

### Train/Eval/QAT

1. Evaluation
  - Execute run_eval.sh.
    ```shell
    sudo -E bash code/run_eval.sh
    ```

2. Training
    ```shell
    sudo -E bash code/run_train.sh
    ```

3. Model quantization
For quantization you need a seperate quant exp file. Please refer to the example. The get_model and get_eveluator functions need to be adapted as shown in the file.
    ```shell
    sudo -E bash code/run_quant.sh
    ```

5. QAT(Quantization-Aware-Training), model converting and xmodel dumping
  - Configure the variables and in `code/run_qat.sh` and `code/exps/example/custom/yolox_nano_deploy_relu_qat.py`, read the steps(including QAT, model testing, model converting and xmodel dumping) in the script and run the step you want.
    ```shell
    sudo -E bash code/run_qat.sh
    ```
5.
After quantization check https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Compiling-for-DPU

For Kv260:

```bash
vai_c_xir -x PATH_TO/YOUR.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o EXPORT_PATH/ -n NEWNAME
```

Export svg graph

```bash
xdputil xmodel Path_TO_/xmodel -s
```

### Performance

|Metric | Float | Quantized | QAT |
| -     | -    | - | - |
|AP0.50:0.95|0.220|0.136|0.210|


### Model_info

1. Data preprocess
    ```
    data channels order: BGR
    keeping the aspect ratio of H/W, resize image with bilinear interpolation to make the long side to be 416, pad the image with (114,114,114) along the height side to get image with shape of (H,W)=(416,416)
    ```

### Acknowledgement

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)
