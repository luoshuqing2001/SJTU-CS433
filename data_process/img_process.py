import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import numpy as np
import onnxruntime as ort

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

ort_session = ort.InferenceSession("./resnet18.onnx")

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

for i in range(1, 10001):
    s = str('%08d'%i)
    path_img = "./ILSVRC2012_val_" + s + ".JPEG"
    print(path_img)
    
    img_rgb = Image.open(path_img).convert('RGB')
    
    img_np = inference_transform(img_rgb).numpy()
    
    img_np = np.expand_dims(img_np, axis=0)
    
    output = ort_session.run(
        None,
        {"input": img_np}
    )
    
    txt_path = "./output_data/" + str(i) + ".txt"
    
    file = open(txt_path, 'w')
    
    np.savetxt(file, np.array(output[0][0]), fmt='%.8f')
    
    # for i in range(3):
    #     input = img_np[i]
    #     np.savetxt(file, input, fmt='%.8f')
        
    file.close()
