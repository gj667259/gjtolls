import numpy as np


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


def non_max_suppression(prediction, conf_thres=0.25, nms_thres=0.45):
    nc = prediction.shape[1] - 4  # number of classes  default = 80
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        image_pred = np.transpose(image_pred)
        x = image_pred[xc[i]] # confidence
        # Detections matrix nx6 (xyxy, conf, cls)
        [box, cls, _] = np.split(x, (4, 4+nc), axis=1)
       
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        conf, j = cls.max(1, keepdims=True), cls.argmax(1, keepdims=True)
        if not x.shape[0]:
            continue
        
        detections = np.concatenate((box, conf, conf, j), 1)
        #   获得预测结果中包含的所有种类
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            # 获得某一类得分筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]
            # 按照存在物体的置信度排序
            detections_class = detections_class[(-detections_class[:,4]).argsort()]
            # 进行非极大抑制
            max_detections = []
            while detections_class.shape[0]:
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                # max_detections.append(detections_class[0].unsqueeze(0))
                max_detections.append(np.array([detections_class[0]]))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠
            max_detections = np.concatenate(max_detections)
            
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))
    return output


def xywh2xyxy(x):
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def run():
    import torch
    load_torch = torch.load('result.pt')[0]
    load_torch = load_torch.transpose(-1, -2)
    t = torch.zeros((1, 8400, 85))
    t[:, :, 0:4] = load_torch[:, :, 0:4]
    t[:, :, 5:85] = load_torch[:, :, 4:84]
    t[:, :, 4], _ = torch.max(load_torch[:, :, 4:84], dim=2)
    n = non_max_suppression(t, num_classes=80)
    print(n)


def t():
    import torch
    load_torch = torch.load('result.pt')[0].numpy()
    n = non_max_suppression(load_torch)
    print(n)


if __name__ == "__main__":
    t()
