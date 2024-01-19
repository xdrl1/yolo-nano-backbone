import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python
def parse_log_file(log_file_path):
    # Regular expression to match the AP line
    ap_regex = r"Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d\.]+)"

    # Lists to store the extracted data
    epochs_loss = []
    iters_loss = []
    total_losses = []
    epochs_ap = []
    ap_values = []

    # Read and parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'yolox.core.trainer:after_iter:253 - epoch:' in line:
                parts = line.split(',')

                # Extract epoch and iter for total loss
                epoch_part = parts[0]
                epoch_loss = int(epoch_part.split('epoch: ')[1].split('/')[0])
                iter_part = parts[1]
                iter_loss = int(iter_part.split('iter: ')[1].split('/')[0])

                # Extract total_loss
                total_loss_part = [p for p in parts if 'total_loss:' in p][0]
                total_loss = float(total_loss_part.split('total_loss: ')[1])

                epochs_loss.append(epoch_loss)
                iters_loss.append(iter_loss)
                total_losses.append(total_loss)
            elif re.search(ap_regex, line):
                # Extract AP value and corresponding epoch
                ap_match = re.search(ap_regex, line)
                if ap_match:
                    ap_values.append(float(ap_match.group(1)))
                    epochs_ap.append(epoch_loss)  # Use the most recent epoch_loss value

    # Create DataFrames with the extracted data
    loss_data = pd.DataFrame({
        'Epoch': epochs_loss,
        'Iter': iters_loss,
        'Total Loss': total_losses
    })

    ap_data = pd.DataFrame({
        'Epoch': epochs_ap,
        'AP': ap_values
    })

    return loss_data, ap_data

# Paths to log files
log_file_path_1 = '../../YOLOX_outputs/yolox_nano_deploy_relu_bird_exp1/train_log_1.txt'
log_file_path_2 = '../../YOLOX_outputs/yolox_nano_deploy_relu_bird/train_log.txt'

# Parse both log files
loss_data_1, ap_data_1 = parse_log_file(log_file_path_1)
loss_data_2, ap_data_2 = parse_log_file(log_file_path_2)

# Plotting Total Loss over Epochs
plt.figure(figsize=(12, 8))
plt.plot(loss_data_1['Epoch'], loss_data_1['Total Loss'], label='Total Loss (First Training)', color='blue')
plt.plot(loss_data_2['Epoch'], loss_data_2['Total Loss'], label='Total Loss (Second Training)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Comparison of Total Loss Over Epochs')
plt.legend()
plt.ylim([0,20])
plt.yticks([i for i in np.arange(0, 20, 0.5)])
plt.savefig("/home/vitis-ai-user/yolonano/code/Comparison/images_2/TotalLossPlot.png")

# Plotting AP over Epochs
plt.figure(figsize=(12, 8))
plt.plot(ap_data_1['Epoch'], ap_data_1['AP'], label='AP (First Training)', color='green')
plt.plot(ap_data_2['Epoch'], ap_data_2['AP'], label='AP (Second Training)', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Average Precision (AP)')
plt.title('Comparison of AP Over Epochs')
plt.legend()
plt.ylim([0,0.3])
plt.yticks([i for i in np.arange(0, 0.31, 0.01)])
plt.savefig("/home/vitis-ai-user/yolonano/code/Comparison/images_2/AP_Plot.png")

