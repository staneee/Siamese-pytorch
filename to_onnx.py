import numpy as np
import onnxoptimizer
from PIL import Image
from nets.siamese import Siamese as siamese
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from siamese import Siamese

if __name__ == "__main__":

    modelPath = "./logs/ep031-loss0.026-val_loss0.028.pth"
    savePath = "./model_data/best.onnx"

    imageChannel=3
    imageWidth=105
    imageHeight=105
    input_shape = [imageChannel, imageWidth,  imageHeight]


    # 加载模型
    device  = torch.device('cpu')
    model   = siamese(input_shape)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    # 生产模型
    model = model.eval()


    # 导出参数
    ## 输入数据结构
    input_shape2 = (imageChannel,  imageWidth, imageHeight)
    dummy_input = (
        torch.randn(1, *input_shape2).to(device),
        torch.randn(1, *input_shape2).to(device)
    )
    ## 输入输出参数
    input_names = ["input1","input2"]
    output_names = ["output"]

    # torch.onnx.export(model, (torch.rand(1, 3, 224, 224).to(device), torch.rand(1, 3, 224, 224).to(device)), args.out_path, input_names=['input'],
    #                   output_names=['output'], export_params=True)
    torch.onnx.export(model,
                      dummy_input,
                      savePath,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      export_params=True,                      
                      keep_initializers_as_inputs=True
                      )    
    
    