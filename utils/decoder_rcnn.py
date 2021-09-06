import cv2 
import matplotlib 
import numpy 

# 関節同士の組み合わせの分だけ座標を求めてイメージとする
# 参照 https://debuggercafe.com/human-pose-detection-using-pytorch-keypoint-rcnn/
edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

def decode_img(outputs, image):
    for ii in range(outputs[0]["scores"].size()[0]): # 認識された人数の分だけループさせる
        if outputs[0]["scores"][ii].item() > 0.9:
            keypoints = outputs[0]["keypoints"][ii].detach().cpu().numpy()
            for i in range(len(keypoints)):
                pos = keypoints[i]
                x = int(pos[0])
                y = int(pos[1])
                cv2.circle(image, (x, y) ,3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            for ie, (e1, e2) in enumerate(edges):
                rgb = matplotlib.colors.hsv_to_rgb([
                                                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb * 255
                x1 = int(keypoints[e1][0])
                y1 = int(keypoints[e1][1])
                x2 = int(keypoints[e2][0])
                y2 = int(keypoints[e2][1])
                cv2.line(image, (x1, y1), (x2, y2), tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return image 