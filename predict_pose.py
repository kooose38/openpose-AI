import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as  plt 
from PIL import Image 
import cv2 
import uuid 
import os, time, json, pickle 
from torchvision import transforms
from torchvision.models import vgg19
import argparse

from networks.openpose import OpenPoseNet 
from utils.decode import decode_pose 
from utils.transform import transform  

parser = argparse.ArgumentParser()
parser.add_argument("arg1", help="image path name", type=str, default="1.jpg")
args = parser.parse_args()
root_path = "img/"


def load_weight(net, filename="./models/pose_model_scratch.pth"):
    weights = torch.load(filename, map_location={"cuda:0": "cpu"})
    keys = list(weights.keys())
    load_w = {}

    for i in range(len(keys)):
        load_w[list(net.state_dict().keys())[i]] = weights[list(keys)[i]]

    state = net.state_dict()
    state.update(load_w)
    net.load_state_dict(state)
    net.eval()
    return net 


def show_img(img):
    fig = plt.figure()
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    os.makedirs("results", exist_ok=True)
    id = uuid.uuid4()
    fig.savefig(f"results/{str(id)[:4]}.png")
    print(f"saving result image for path results/{str(id)[:4]}.png ")

    
def detect(net, img_tensor):
    with torch.no_grad():
        output = net(img_tensor)
        heatmap = output[-1][0].detach().cpu().numpy().transpose(1, 2, 0)
        pafs = output[-2][0].detach().cpu().numpy().transpose(1, 2, 0)
    return heatmap, pafs 


def main(img_path: str):
    # モデルの読み込み
    net = OpenPoseNet()
    net = load_weight(net)
    # 画像の前処理
    img = Image.open(img_path)
    oriImg = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)
    # 推論
    heatmap, pafs = detect(net, img_tensor)
    _, result_img, _, _ = decode_pose(oriImg, heatmap, pafs)
    # 画像の表示
    show_img(result_img)
    
if __main__ = "__name__":
    main(root_path+args.arg1)