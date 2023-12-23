# SJTU-CS433
SJTU CS433 Parallel and Distributed Programming Final Project

ResNet18 implementation and TensorCore simulator
## 文件目录结构
```
.\SJTU-CS433
│   document.pdf
│   Group_1.pptx
|   1_Tensor_Core的使用与原理实现_.pdf
│   
├───data_process（数据处理）
│       forward.py（计算中间层准确输出）
│       img_process.py（图片处理）
│       process.py（读取模型参数）
│       sass指令文件分析.md
│       sass指令文件分析.pdf
│       
├───general_data（请在 https://jbox.sjtu.edu.cn/l/S1IbkL 中下载）
│   ├───data_1（实验一结果）
│   │   ├───result
│   │   └───time
│   ├───data_2（实验二结果）
│   │   ├───result
│   │   └───time
│   ├───input_data（一万个 3 × 224 × 224 图片张量）
│   ├───model_data（onnx 模型参数）
│   └───result_b（baseline 计算结果）
|
├───ResNet_1
│   │   main.cpp
│   │   Makefile
│   │   
│   ├───bin
│   ├───data
│   │   ├───input_data（请从 general_data 中复制）
│   │   ├───model_data（请从 general_data 中复制）
│   │   ├───result
│   │   └───time
│   ├───include
│   │       cuda_kernel.cuh
│   │       
│   └───src
│           cuda_kernel.cu
│           
└───ResNet_2
    │   Makefile
    │   resnet_test.cpp
    │   tensor_core.h
    │   
    └───data
        ├───input_data（请从 general_data 中复制）
        ├───model_data（请从 general_data 中复制）
        ├───result
        └───time
```
