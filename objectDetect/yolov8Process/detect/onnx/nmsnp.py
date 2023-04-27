import numpy as np
import re
import yaml


def yaml_load(file='data.yaml', append_filename=False):
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * \
                np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / np.clip(b1_area + b2_area - inter_area, a_min = 1e-6, a_max=None)
    return iou


def nms(prediction, conf_thres=0.25, nms_thres=0.45):
    nc = prediction.shape[1] - 4  
    mi = 4 + nc  
    xc = prediction[:, 4:mi].max(1) > conf_thres  
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        image_pred = np.transpose(image_pred)
        x = image_pred[xc[i]] 
        [box, cls, _] = np.split(x, (4, 4+nc), axis=1)
        box = xywh2xyxy(box)  
        conf, j = cls.max(1, keepdims=True), cls.argmax(1, keepdims=True)
        if not x.shape[0]:
            continue
        detections = np.concatenate((box, conf, conf, j), 1)
        unique_labels = np.unique(detections[:, -1])
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            detections_class = detections_class[(-detections_class[:,4]).argsort()]        
            max_detections = []
            while detections_class.shape[0]:
                max_detections.append(np.array([detections_class[0]]))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            max_detections = np.concatenate(max_detections)
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))
    return output


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  
    y[..., 1] = x[..., 1] - x[..., 3] / 2  
    y[..., 2] = x[..., 0] + x[..., 2] / 2  
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def run():
    import torch
    load_torch = torch.load('ggg.pt')[0]
    load_torch = load_torch.transpose(-1, -2)
    t = torch.zeros((1, 8400, 85))
    t[:, :, 0:4] = load_torch[:, :, 0:4]
    t[:, :, 5:85] = load_torch[:, :, 4:84]
    t[:, :, 4], _ = torch.max(load_torch[:, :, 4:84], dim=2)
    n = non_max_suppression(t, num_classes=80)
    print(n)


def t():
    import torch
    load_torch = torch.load('ggg.pt')[0].numpy()
    n = non_max_suppression(load_torch)
    print(n)


if __name__ == "__main__":
    t()
