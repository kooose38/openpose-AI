import torch , torchvision 
from torchvision import transforms 

class Transform():
    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose([
                                        transforms.Resize((resize, resize)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)
        ])  
    def __call__(self, img):
        return self.data_transform(img)
    
    
class TransformRCNN():
    def __init__(self):
        self.data_transform = transforms.Compose([
                                                  transforms.ToTensor()
        ])
    def __call__(self, img):
        return self.data_transform(img)

    
RESIZE = 368
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
transform = Transform(RESIZE, MEAN, STD)
transform_rcnn = TransformRCNN()