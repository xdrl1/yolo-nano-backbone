 export W_QUANT=1
 export CUDA_HOME=/usr/local/cuda

#modify for qat

# export CUDA_VISIBLE_DEVICES=0
# GPU_NUM=1
# BATCH=8
CFG=code/exps/example/custom/yolox_nano_deploy_relu_qat_building.py
# Step1: QAT
# /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m yolox.tools.train -f ${CFG} -d ${GPU_NUM} -b ${BATCH} -o

#modify for evaluation

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
BATCH=32

WORKSPACE=YOLOX_outputs/yolox_nano_deploy_relu_qat_building # path to best .pth from QAT


#Step2: Eval accuracy after QAT
#QAT_WEIGHTS=${WORKSPACE}/best_ckpt_qat.pth  # assign the path to your QAT weights
#/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} -b ${BATCH} -d ${GPU_NUM}

# Step3: Convert the QAT model to deployble model and verify the accuracy
#CVT_DIR=${WORKSPACE}/convert_qat_results
#QAT_WEIGHTS=${WORKSPACE}/best_ckpt_qat.pth
#/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} --cvt_dir ${CVT_DIR} -b ${BATCH} -d ${GPU_NUM}

# Step4: Dump xmodel
QAT_WEIGHTS=${WORKSPACE}/best_ckpt_qat.pth
CVT_DIR=${WORKSPACE}/convert_qat_results
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} --cvt_dir ${CVT_DIR} --is_dump
