# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class YOLOX(torch.nn.Module):
    def __init__(self):
        super(YOLOX, self).__init__()
        self.module_0 = py_nndct.nn.Input() #YOLOX::input_0
        self.module_1 = py_nndct.nn.quant_input() #YOLOX::YOLOX/QuantStub[quant_in]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/BaseConv[stem]/Conv2d[conv]/input.3
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/BaseConv[stem]/ReLU[act]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=16, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.9
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.13
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.15
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.19
        self.module_8 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.21
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv1]/ReLU[act]/input.29
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.25
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv2]/ReLU[act]/22670
        self.module_12 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.31
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.35
        self.module_14 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=16, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.37
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.41
        self.module_16 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.43
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/22748
        self.module_18 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/22750
        self.module_19 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/Cat[cat]/input.47
        self.module_20 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.49
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark2]/Q_CSPLayer[1]/BaseConv[conv3]/ReLU[act]/input.53
        self.module_22 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.55
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.59
        self.module_24 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.61
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.65
        self.module_26 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.67
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv1]/ReLU[act]/input.75
        self.module_28 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.71
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv2]/ReLU[act]/22883
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.77
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.81
        self.module_32 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.83
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.87
        self.module_34 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.89
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/22961
        self.module_36 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/input.93
        self.module_37 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.95
        self.module_38 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/ReLU[act]/input.99
        self.module_39 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.101
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.105
        self.module_41 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.107
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23041
        self.module_43 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/Add[sc_add]/input.111
        self.module_44 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/Conv2d[conv]/input.113
        self.module_45 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/ReLU[act]/input.117
        self.module_46 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.119
        self.module_47 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.123
        self.module_48 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.125
        self.module_49 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23121
        self.module_50 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/Add[sc_add]/23123
        self.module_51 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/Cat[cat]/input.129
        self.module_52 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.131
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark3]/Q_CSPLayer[1]/BaseConv[conv3]/ReLU[act]/input.135
        self.module_54 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.137
        self.module_55 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.141
        self.module_56 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.143
        self.module_57 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.147
        self.module_58 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv1]/Conv2d[conv]/input.149
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv1]/ReLU[act]/input.157
        self.module_60 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv2]/Conv2d[conv]/input.153
        self.module_61 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv2]/ReLU[act]/23256
        self.module_62 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.159
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.163
        self.module_64 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.165
        self.module_65 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.169
        self.module_66 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.171
        self.module_67 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23334
        self.module_68 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[0]/Add[sc_add]/input.175
        self.module_69 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.177
        self.module_70 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/BaseConv[conv1]/ReLU[act]/input.181
        self.module_71 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.183
        self.module_72 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.187
        self.module_73 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.189
        self.module_74 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23414
        self.module_75 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[1]/Add[sc_add]/input.193
        self.module_76 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/Conv2d[conv]/input.195
        self.module_77 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/BaseConv[conv1]/ReLU[act]/input.199
        self.module_78 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.201
        self.module_79 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.205
        self.module_80 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.207
        self.module_81 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23494
        self.module_82 = py_nndct.nn.Add() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Sequential[m]/Q_Bottleneck[2]/Add[sc_add]/23496
        self.module_83 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/Cat[cat]/input.211
        self.module_84 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv3]/Conv2d[conv]/input.213
        self.module_85 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark4]/Q_CSPLayer[1]/BaseConv[conv3]/ReLU[act]/input.217
        self.module_86 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.219
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.223
        self.module_88 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.225
        self.module_89 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.229
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv1]/Conv2d[conv]/input.231
        self.module_91 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv1]/ReLU[act]/23603
        self.module_92 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[0]/23617
        self.module_93 = py_nndct.nn.MaxPool2d(kernel_size=[9, 9], stride=[1, 1], padding=[4, 4], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[1]/23631
        self.module_94 = py_nndct.nn.MaxPool2d(kernel_size=[13, 13], stride=[1, 1], padding=[6, 6], dilation=[1, 1], ceil_mode=False) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/MaxPool2d[m]/ModuleList[2]/23645
        self.module_95 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/Cat[cat]/input.235
        self.module_96 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv2]/Conv2d[conv]/input.237
        self.module_97 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_SPPBottleneck[1]/BaseConv[conv2]/ReLU[act]/input.241
        self.module_98 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv1]/Conv2d[conv]/input.243
        self.module_99 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv1]/ReLU[act]/input.251
        self.module_100 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv2]/Conv2d[conv]/input.247
        self.module_101 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv2]/ReLU[act]/23726
        self.module_102 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.253
        self.module_103 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.257
        self.module_104 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.259
        self.module_105 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.263
        self.module_106 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.265
        self.module_107 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/23804
        self.module_108 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/Cat[cat]/input.269
        self.module_109 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv3]/Conv2d[conv]/input.271
        self.module_110 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/CSPDarknet[backbone]/Sequential[dark5]/Q_CSPLayer[2]/BaseConv[conv3]/ReLU[act]/input.275
        self.module_111 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[lateral_conv0]/Conv2d[conv]/input.277
        self.module_112 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[lateral_conv0]/ReLU[act]/input.281
        self.module_113 = py_nndct.nn.Interpolate() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Upsample[upsample]/23868
        self.module_114 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_f0]/input.283
        self.module_115 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv1]/Conv2d[conv]/input.285
        self.module_116 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv1]/ReLU[act]/input.293
        self.module_117 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv2]/Conv2d[conv]/input.289
        self.module_118 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv2]/ReLU[act]/23923
        self.module_119 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.295
        self.module_120 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.299
        self.module_121 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.301
        self.module_122 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.305
        self.module_123 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.307
        self.module_124 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/24001
        self.module_125 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/Cat[cat]/input.311
        self.module_126 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv3]/Conv2d[conv]/input.313
        self.module_127 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p4]/BaseConv[conv3]/ReLU[act]/input.317
        self.module_128 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[reduce_conv1]/Conv2d[conv]/input.319
        self.module_129 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/BaseConv[reduce_conv1]/ReLU[act]/input.323
        self.module_130 = py_nndct.nn.Interpolate() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Upsample[upsample]/24061
        self.module_131 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_f1]/input.325
        self.module_132 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv1]/Conv2d[conv]/input.327
        self.module_133 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv1]/ReLU[act]/input.335
        self.module_134 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv2]/Conv2d[conv]/input.331
        self.module_135 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv2]/ReLU[act]/24116
        self.module_136 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.337
        self.module_137 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.341
        self.module_138 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.343
        self.module_139 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.347
        self.module_140 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.349
        self.module_141 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/24194
        self.module_142 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/Cat[cat]/input.353
        self.module_143 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv3]/Conv2d[conv]/input.355
        self.module_144 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_p3]/BaseConv[conv3]/ReLU[act]/input.359
        self.module_145 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv2]/BaseConv[dconv]/Conv2d[conv]/input.361
        self.module_146 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv2]/BaseConv[dconv]/ReLU[act]/input.365
        self.module_147 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv2]/BaseConv[pconv]/Conv2d[conv]/input.367
        self.module_148 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv2]/BaseConv[pconv]/ReLU[act]/24275
        self.module_149 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_p1]/input.371
        self.module_150 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv1]/Conv2d[conv]/input.373
        self.module_151 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv1]/ReLU[act]/input.381
        self.module_152 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv2]/Conv2d[conv]/input.377
        self.module_153 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv2]/ReLU[act]/24330
        self.module_154 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.383
        self.module_155 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.387
        self.module_156 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.389
        self.module_157 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.393
        self.module_158 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.395
        self.module_159 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/24408
        self.module_160 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/Cat[cat]/input.399
        self.module_161 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv3]/Conv2d[conv]/input.401
        self.module_162 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n3]/BaseConv[conv3]/ReLU[act]/input.405
        self.module_163 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv1]/BaseConv[dconv]/Conv2d[conv]/input.407
        self.module_164 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv1]/BaseConv[dconv]/ReLU[act]/input.411
        self.module_165 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv1]/BaseConv[pconv]/Conv2d[conv]/input.413
        self.module_166 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/DWConv[bu_conv1]/BaseConv[pconv]/ReLU[act]/24489
        self.module_167 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Cat[cat_p0]/input.417
        self.module_168 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv1]/Conv2d[conv]/input.419
        self.module_169 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv1]/ReLU[act]/input.427
        self.module_170 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv2]/Conv2d[conv]/input.423
        self.module_171 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv2]/ReLU[act]/24544
        self.module_172 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/Conv2d[conv]/input.429
        self.module_173 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/BaseConv[conv1]/ReLU[act]/input.433
        self.module_174 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/Conv2d[conv]/input.435
        self.module_175 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[dconv]/ReLU[act]/input.439
        self.module_176 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/Conv2d[conv]/input.441
        self.module_177 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Sequential[m]/Q_Bottleneck[0]/DWConv[conv2]/BaseConv[pconv]/ReLU[act]/24622
        self.module_178 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/Cat[cat]/input.445
        self.module_179 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv3]/Conv2d[conv]/input.447
        self.module_180 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOPAFPN[backbone]/Q_CSPLayer[C3_n4]/BaseConv[conv3]/ReLU[act]/input.559
        self.module_181 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[0]/Conv2d[conv]/input.451
        self.module_182 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[0]/ReLU[act]/input.455
        self.module_183 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.457
        self.module_184 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.461
        self.module_185 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.463
        self.module_186 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.467
        self.module_187 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.469
        self.module_188 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.473
        self.module_189 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.475
        self.module_190 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[0]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input.479
        self.module_191 = py_nndct.nn.Conv2d(in_channels=64, out_channels=80, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[0]/24804
        self.module_192 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.481
        self.module_193 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.485
        self.module_194 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.487
        self.module_195 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.491
        self.module_196 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.493
        self.module_197 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.497
        self.module_198 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.499
        self.module_199 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[0]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input.503
        self.module_200 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[0]/24927
        self.module_201 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[0]/24946
        self.module_202 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[0]/inputs.3
        self.module_203 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[0]/24950
        self.module_204 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[1]/Conv2d[conv]/input.505
        self.module_205 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[1]/ReLU[act]/input.509
        self.module_206 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.511
        self.module_207 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.515
        self.module_208 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.517
        self.module_209 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.521
        self.module_210 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.523
        self.module_211 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.527
        self.module_212 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.529
        self.module_213 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[1]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input.533
        self.module_214 = py_nndct.nn.Conv2d(in_channels=64, out_channels=80, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[1]/25101
        self.module_215 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.535
        self.module_216 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.539
        self.module_217 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.541
        self.module_218 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.545
        self.module_219 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.547
        self.module_220 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.551
        self.module_221 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.553
        self.module_222 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[1]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input.557
        self.module_223 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[1]/25224
        self.module_224 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[1]/25243
        self.module_225 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[1]/inputs.5
        self.module_226 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[1]/25247
        self.module_227 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[2]/Conv2d[conv]/input.561
        self.module_228 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/BaseConv[stems]/ModuleList[2]/ReLU[act]/input.565
        self.module_229 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.567
        self.module_230 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.571
        self.module_231 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.573
        self.module_232 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.577
        self.module_233 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.579
        self.module_234 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.583
        self.module_235 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.585
        self.module_236 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[cls_convs]/ModuleList[2]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input.589
        self.module_237 = py_nndct.nn.Conv2d(in_channels=64, out_channels=80, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[cls_preds]/ModuleList[2]/25398
        self.module_238 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[0]/BaseConv[dconv]/Conv2d[conv]/input.591
        self.module_239 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[0]/BaseConv[dconv]/ReLU[act]/input.595
        self.module_240 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[0]/BaseConv[pconv]/Conv2d[conv]/input.597
        self.module_241 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[0]/BaseConv[pconv]/ReLU[act]/input.601
        self.module_242 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[1]/BaseConv[dconv]/Conv2d[conv]/input.603
        self.module_243 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[1]/BaseConv[dconv]/ReLU[act]/input.607
        self.module_244 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[1]/BaseConv[pconv]/Conv2d[conv]/input.609
        self.module_245 = py_nndct.nn.ReLU(inplace=True) #YOLOX::YOLOX/YOLOXHead[head]/Sequential[reg_convs]/ModuleList[2]/DWConv[1]/BaseConv[pconv]/ReLU[act]/input
        self.module_246 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[reg_preds]/ModuleList[2]/25521
        self.module_247 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLOX::YOLOX/YOLOXHead[head]/Conv2d[obj_preds]/ModuleList[2]/25540
        self.module_248 = py_nndct.nn.Cat() #YOLOX::YOLOX/YOLOXHead[head]/Cat[cat_list]/ModuleList[2]/inputs
        self.module_249 = py_nndct.nn.dequant_output() #YOLOX::YOLOX/YOLOXHead[head]/DeQuantStub[quant_outs]/ModuleList[2]/25544

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_8 = self.module_8(output_module_0)
        output_module_8 = self.module_9(output_module_8)
        output_module_10 = self.module_10(output_module_0)
        output_module_10 = self.module_11(output_module_10)
        output_module_12 = self.module_12(output_module_8)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_16(output_module_12)
        output_module_12 = self.module_17(output_module_12)
        output_module_12 = self.module_18(input=output_module_12, other=output_module_8, alpha=1)
        output_module_12 = self.module_19(dim=1, tensors=[output_module_12,output_module_10])
        output_module_12 = self.module_20(output_module_12)
        output_module_12 = self.module_21(output_module_12)
        output_module_12 = self.module_22(output_module_12)
        output_module_12 = self.module_23(output_module_12)
        output_module_12 = self.module_24(output_module_12)
        output_module_12 = self.module_25(output_module_12)
        output_module_26 = self.module_26(output_module_12)
        output_module_26 = self.module_27(output_module_26)
        output_module_28 = self.module_28(output_module_12)
        output_module_28 = self.module_29(output_module_28)
        output_module_30 = self.module_30(output_module_26)
        output_module_30 = self.module_31(output_module_30)
        output_module_30 = self.module_32(output_module_30)
        output_module_30 = self.module_33(output_module_30)
        output_module_30 = self.module_34(output_module_30)
        output_module_30 = self.module_35(output_module_30)
        output_module_30 = self.module_36(input=output_module_30, other=output_module_26, alpha=1)
        output_module_37 = self.module_37(output_module_30)
        output_module_37 = self.module_38(output_module_37)
        output_module_37 = self.module_39(output_module_37)
        output_module_37 = self.module_40(output_module_37)
        output_module_37 = self.module_41(output_module_37)
        output_module_37 = self.module_42(output_module_37)
        output_module_37 = self.module_43(input=output_module_37, other=output_module_30, alpha=1)
        output_module_44 = self.module_44(output_module_37)
        output_module_44 = self.module_45(output_module_44)
        output_module_44 = self.module_46(output_module_44)
        output_module_44 = self.module_47(output_module_44)
        output_module_44 = self.module_48(output_module_44)
        output_module_44 = self.module_49(output_module_44)
        output_module_44 = self.module_50(input=output_module_44, other=output_module_37, alpha=1)
        output_module_44 = self.module_51(dim=1, tensors=[output_module_44,output_module_28])
        output_module_44 = self.module_52(output_module_44)
        output_module_44 = self.module_53(output_module_44)
        output_module_54 = self.module_54(output_module_44)
        output_module_54 = self.module_55(output_module_54)
        output_module_54 = self.module_56(output_module_54)
        output_module_54 = self.module_57(output_module_54)
        output_module_58 = self.module_58(output_module_54)
        output_module_58 = self.module_59(output_module_58)
        output_module_60 = self.module_60(output_module_54)
        output_module_60 = self.module_61(output_module_60)
        output_module_62 = self.module_62(output_module_58)
        output_module_62 = self.module_63(output_module_62)
        output_module_62 = self.module_64(output_module_62)
        output_module_62 = self.module_65(output_module_62)
        output_module_62 = self.module_66(output_module_62)
        output_module_62 = self.module_67(output_module_62)
        output_module_62 = self.module_68(input=output_module_62, other=output_module_58, alpha=1)
        output_module_69 = self.module_69(output_module_62)
        output_module_69 = self.module_70(output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_69 = self.module_72(output_module_69)
        output_module_69 = self.module_73(output_module_69)
        output_module_69 = self.module_74(output_module_69)
        output_module_69 = self.module_75(input=output_module_69, other=output_module_62, alpha=1)
        output_module_76 = self.module_76(output_module_69)
        output_module_76 = self.module_77(output_module_76)
        output_module_76 = self.module_78(output_module_76)
        output_module_76 = self.module_79(output_module_76)
        output_module_76 = self.module_80(output_module_76)
        output_module_76 = self.module_81(output_module_76)
        output_module_76 = self.module_82(input=output_module_76, other=output_module_69, alpha=1)
        output_module_76 = self.module_83(dim=1, tensors=[output_module_76,output_module_60])
        output_module_76 = self.module_84(output_module_76)
        output_module_76 = self.module_85(output_module_76)
        output_module_86 = self.module_86(output_module_76)
        output_module_86 = self.module_87(output_module_86)
        output_module_86 = self.module_88(output_module_86)
        output_module_86 = self.module_89(output_module_86)
        output_module_86 = self.module_90(output_module_86)
        output_module_86 = self.module_91(output_module_86)
        output_module_92 = self.module_92(output_module_86)
        output_module_93 = self.module_93(output_module_86)
        output_module_94 = self.module_94(output_module_86)
        output_module_95 = self.module_95(dim=1, tensors=[output_module_86,output_module_92,output_module_93,output_module_94])
        output_module_95 = self.module_96(output_module_95)
        output_module_95 = self.module_97(output_module_95)
        output_module_98 = self.module_98(output_module_95)
        output_module_98 = self.module_99(output_module_98)
        output_module_100 = self.module_100(output_module_95)
        output_module_100 = self.module_101(output_module_100)
        output_module_98 = self.module_102(output_module_98)
        output_module_98 = self.module_103(output_module_98)
        output_module_98 = self.module_104(output_module_98)
        output_module_98 = self.module_105(output_module_98)
        output_module_98 = self.module_106(output_module_98)
        output_module_98 = self.module_107(output_module_98)
        output_module_98 = self.module_108(dim=1, tensors=[output_module_98,output_module_100])
        output_module_98 = self.module_109(output_module_98)
        output_module_98 = self.module_110(output_module_98)
        output_module_98 = self.module_111(output_module_98)
        output_module_98 = self.module_112(output_module_98)
        output_module_113 = self.module_113(input=output_module_98, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_113 = self.module_114(dim=1, tensors=[output_module_113,output_module_76])
        output_module_115 = self.module_115(output_module_113)
        output_module_115 = self.module_116(output_module_115)
        output_module_117 = self.module_117(output_module_113)
        output_module_117 = self.module_118(output_module_117)
        output_module_115 = self.module_119(output_module_115)
        output_module_115 = self.module_120(output_module_115)
        output_module_115 = self.module_121(output_module_115)
        output_module_115 = self.module_122(output_module_115)
        output_module_115 = self.module_123(output_module_115)
        output_module_115 = self.module_124(output_module_115)
        output_module_115 = self.module_125(dim=1, tensors=[output_module_115,output_module_117])
        output_module_115 = self.module_126(output_module_115)
        output_module_115 = self.module_127(output_module_115)
        output_module_115 = self.module_128(output_module_115)
        output_module_115 = self.module_129(output_module_115)
        output_module_130 = self.module_130(input=output_module_115, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_130 = self.module_131(dim=1, tensors=[output_module_130,output_module_44])
        output_module_132 = self.module_132(output_module_130)
        output_module_132 = self.module_133(output_module_132)
        output_module_134 = self.module_134(output_module_130)
        output_module_134 = self.module_135(output_module_134)
        output_module_132 = self.module_136(output_module_132)
        output_module_132 = self.module_137(output_module_132)
        output_module_132 = self.module_138(output_module_132)
        output_module_132 = self.module_139(output_module_132)
        output_module_132 = self.module_140(output_module_132)
        output_module_132 = self.module_141(output_module_132)
        output_module_132 = self.module_142(dim=1, tensors=[output_module_132,output_module_134])
        output_module_132 = self.module_143(output_module_132)
        output_module_132 = self.module_144(output_module_132)
        output_module_145 = self.module_145(output_module_132)
        output_module_145 = self.module_146(output_module_145)
        output_module_145 = self.module_147(output_module_145)
        output_module_145 = self.module_148(output_module_145)
        output_module_145 = self.module_149(dim=1, tensors=[output_module_145,output_module_115])
        output_module_150 = self.module_150(output_module_145)
        output_module_150 = self.module_151(output_module_150)
        output_module_152 = self.module_152(output_module_145)
        output_module_152 = self.module_153(output_module_152)
        output_module_150 = self.module_154(output_module_150)
        output_module_150 = self.module_155(output_module_150)
        output_module_150 = self.module_156(output_module_150)
        output_module_150 = self.module_157(output_module_150)
        output_module_150 = self.module_158(output_module_150)
        output_module_150 = self.module_159(output_module_150)
        output_module_150 = self.module_160(dim=1, tensors=[output_module_150,output_module_152])
        output_module_150 = self.module_161(output_module_150)
        output_module_150 = self.module_162(output_module_150)
        output_module_163 = self.module_163(output_module_150)
        output_module_163 = self.module_164(output_module_163)
        output_module_163 = self.module_165(output_module_163)
        output_module_163 = self.module_166(output_module_163)
        output_module_163 = self.module_167(dim=1, tensors=[output_module_163,output_module_98])
        output_module_168 = self.module_168(output_module_163)
        output_module_168 = self.module_169(output_module_168)
        output_module_170 = self.module_170(output_module_163)
        output_module_170 = self.module_171(output_module_170)
        output_module_168 = self.module_172(output_module_168)
        output_module_168 = self.module_173(output_module_168)
        output_module_168 = self.module_174(output_module_168)
        output_module_168 = self.module_175(output_module_168)
        output_module_168 = self.module_176(output_module_168)
        output_module_168 = self.module_177(output_module_168)
        output_module_168 = self.module_178(dim=1, tensors=[output_module_168,output_module_170])
        output_module_168 = self.module_179(output_module_168)
        output_module_168 = self.module_180(output_module_168)
        output_module_181 = self.module_181(output_module_132)
        output_module_181 = self.module_182(output_module_181)
        output_module_183 = self.module_183(output_module_181)
        output_module_183 = self.module_184(output_module_183)
        output_module_183 = self.module_185(output_module_183)
        output_module_183 = self.module_186(output_module_183)
        output_module_183 = self.module_187(output_module_183)
        output_module_183 = self.module_188(output_module_183)
        output_module_183 = self.module_189(output_module_183)
        output_module_183 = self.module_190(output_module_183)
        output_module_183 = self.module_191(output_module_183)
        output_module_192 = self.module_192(output_module_181)
        output_module_192 = self.module_193(output_module_192)
        output_module_192 = self.module_194(output_module_192)
        output_module_192 = self.module_195(output_module_192)
        output_module_192 = self.module_196(output_module_192)
        output_module_192 = self.module_197(output_module_192)
        output_module_192 = self.module_198(output_module_192)
        output_module_192 = self.module_199(output_module_192)
        output_module_200 = self.module_200(output_module_192)
        output_module_201 = self.module_201(output_module_192)
        output_module_200 = self.module_202(dim=1, tensors=[output_module_200,output_module_201,output_module_183])
        output_module_200 = self.module_203(input=output_module_200)
        output_module_204 = self.module_204(output_module_150)
        output_module_204 = self.module_205(output_module_204)
        output_module_206 = self.module_206(output_module_204)
        output_module_206 = self.module_207(output_module_206)
        output_module_206 = self.module_208(output_module_206)
        output_module_206 = self.module_209(output_module_206)
        output_module_206 = self.module_210(output_module_206)
        output_module_206 = self.module_211(output_module_206)
        output_module_206 = self.module_212(output_module_206)
        output_module_206 = self.module_213(output_module_206)
        output_module_206 = self.module_214(output_module_206)
        output_module_215 = self.module_215(output_module_204)
        output_module_215 = self.module_216(output_module_215)
        output_module_215 = self.module_217(output_module_215)
        output_module_215 = self.module_218(output_module_215)
        output_module_215 = self.module_219(output_module_215)
        output_module_215 = self.module_220(output_module_215)
        output_module_215 = self.module_221(output_module_215)
        output_module_215 = self.module_222(output_module_215)
        output_module_223 = self.module_223(output_module_215)
        output_module_224 = self.module_224(output_module_215)
        output_module_223 = self.module_225(dim=1, tensors=[output_module_223,output_module_224,output_module_206])
        output_module_223 = self.module_226(input=output_module_223)
        output_module_168 = self.module_227(output_module_168)
        output_module_168 = self.module_228(output_module_168)
        output_module_229 = self.module_229(output_module_168)
        output_module_229 = self.module_230(output_module_229)
        output_module_229 = self.module_231(output_module_229)
        output_module_229 = self.module_232(output_module_229)
        output_module_229 = self.module_233(output_module_229)
        output_module_229 = self.module_234(output_module_229)
        output_module_229 = self.module_235(output_module_229)
        output_module_229 = self.module_236(output_module_229)
        output_module_229 = self.module_237(output_module_229)
        output_module_238 = self.module_238(output_module_168)
        output_module_238 = self.module_239(output_module_238)
        output_module_238 = self.module_240(output_module_238)
        output_module_238 = self.module_241(output_module_238)
        output_module_238 = self.module_242(output_module_238)
        output_module_238 = self.module_243(output_module_238)
        output_module_238 = self.module_244(output_module_238)
        output_module_238 = self.module_245(output_module_238)
        output_module_246 = self.module_246(output_module_238)
        output_module_247 = self.module_247(output_module_238)
        output_module_246 = self.module_248(dim=1, tensors=[output_module_246,output_module_247,output_module_229])
        output_module_246 = self.module_249(input=output_module_246)
        return (output_module_200,output_module_223,output_module_246)
