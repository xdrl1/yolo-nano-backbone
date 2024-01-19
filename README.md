# Manual for KV260

## Requirements

Micro SD-card with Operating system
and Compiled xmodel

## Startup of KV260 board

Place the micro SD-card in the slot and power up the board. Connect a keyboard, mouse and the USB webcam to the board. Clone the bird detection repository to the board by executing the following command in a Terminal:

## Clone Repository
```
cd home/root/
git clone -b kv260 https://github.com/xdrl1/yolo-nano-backbone.git
```
## Install dependicies

You can always check the following steps again by executing the following command:
```
cat README.md
```
```
python -m pip install --upgrade pip wheel setuptools
cd ~/yolo-nano-backbone/
python -m pip install --user -r requirements_board.txt
```
## Run the following to execute the code and detect birds!
```
cd/code/xmodel_inference
python camery.py
```
