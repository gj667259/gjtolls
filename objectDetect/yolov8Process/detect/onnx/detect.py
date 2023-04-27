import onnxruntime as ort
import numpy as np
import cv2
import nmsnp


def run():    
    img = cv2.imread('./bus.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, cha = img_rgb.shape
    length = max(height, width)
    imageB = np.zeros((640, 640, 3), np.float32)
    scale = 640/length
  
    resized_img = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
    h, w, c = resized_img.shape
    imageB[0: h, 0: w] = resized_img

    img_np = np.array([resized_img.transpose(2, 0, 1)])
    img_np /= 255

    m = ort.InferenceSession("yolov8s.onnx",  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    output = m.run(None, {"images": img_np})
    out = output[0]
    res = nmsnp.nms(prediction=out)[0]
    res[:, 0:4] *= scale
    res = res.astype(np.int32)
    return res


if __name__ == '__main__':
    run()
