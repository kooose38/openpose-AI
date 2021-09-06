import cv2 
import matplotlib.pyplot as plt 
import torch, torchvision 
from torchvision import transforms 
import numpy as np 
import os, time, json, uuid 
import matplotlib 
import argparse 
from PIL import Image 

from utils.transform import transform_rcnn 
from utils.decoder_rcnn import decode_img


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument("arg1", help="image path name", type=str, default="1.jpg")
args = parser.parse_args()
root_path = "img/"


def load_model():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
    model.to(device)
    model.eval()
    return model 

def show_img(outputs, image):
    image = decode_img(outputs, image)
    fig = plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    os.makedirs("results", exist_ok=True)
    id = uuid.uuid4()
    fig.savefig(f"results/{str(id)[:4]}.png")
    print(f"saving resutl image for path results/{str(id)[:4]}.png")

def main(img_path: str):
    # モデルの読み込み
    model = load_model()
    # 前処理と表示する画像の調整
    img = Image.open(img_path)
    orgImg = np.array(img, dtype=np.float32)
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_RGB2BGR) / 255.
    img_tensor = transform_rcnn(img).unsqueeze(0)
    # 推論
    with torch.no_grad():
        outputs = model(img_tensor)
    show_img(outputs, orgImg)

if __name__ = "__main__":
    main(root_path+args.arg1)
