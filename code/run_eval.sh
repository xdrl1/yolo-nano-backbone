# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Conducting testing..."
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1

export LD_LIBRARY_PATH=${PWD}/code:${LD_LIBRARY_PATH}
BATCH=32


CFG=code/exps/example/custom/yolox_nano_deploy_relu_building.py
CKPT=float/baseline.pth
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m yolox.tools.eval -f ${CFG} -c ${CKPT} -b ${BATCH} -d ${GPU_NUM}
