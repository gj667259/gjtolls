import onnxruntime as ort
import numpy as np
import cv2

# import torch
import nmsnp

from time import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run():
    # lab = nmsnp.yaml_load('coco128.yaml')
    # print(list(lab['names'].keys()))
    # ll = list(lab['names'].values())
    # ll = list(lab['names'].keys())
    
    img = cv2.imread('./v.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, cha = img_rgb.shape
    length = max(height, width)
    imageB = np.zeros((640, 640, 3), np.float32)
    scale = 640/length
    
    resized_img = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
    h, w, c = resized_img.shape
    imageB[0: h, 0: w] = resized_img

    # cv2.imshow('enhanced', resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    img_np = np.array([resized_img.transpose(2, 0, 1)])
    img_np /= 255

    yolose = ort.InferenceSession("tire.onnx",  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # print(session2.get_providers())

    output = yolose.run(None, {"images": img_np})

    # output = np.array(output)
    # output = torch.tensor(output)
    # torch.save(output, 'ggg.pt')

    out = output[0]

    a = nmsnp.non_max_suppression(prediction=out)
    a = a[0][0]
    a[0:4] *= scale

    print(a)
    a = a.astype(np.int32)
    cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 0, 255), 2)
    # resized_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("enhanced",0)
    cv2.resizeWindow("enhanced", 640, 640)
    cv2.imshow('enhanced', img)
    cv2.waitKey(0)
    # cv2.imwrite('vv.jpg', img) 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
