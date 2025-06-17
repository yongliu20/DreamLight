coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
    'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
    'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
    'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
    'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged',
    'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
    'paper-merged', 'food-other-merged', 'building-other-merged',
    'rock-merged', 'wall-other-merged', 'rug-merged'
]
thing_classes=[
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
stuff_classes=[
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
         'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
         'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
         'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
         'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
         'wall-wood', 'water-other', 'window-blind', 'window-other',
         'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
         'cabinet-merged', 'table-merged', 'floor-other-merged',
         'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
         'paper-merged', 'food-other-merged', 'building-other-merged',
         'rock-merged', 'wall-other-merged', 'rug-merged'
]

import torch
def mask2bbox(masks):
    """Obtain tight bounding boxes of binary masks.

    Args:
        masks (Tensor): Binary mask of shape (n, h, w).

    Returns:
        Tensor: Bboxe with shape (n, 4) of \
            positive region in binary mask.
    """
    N = masks.shape[0]
    bboxes = masks.new_zeros((N, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(N):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1])

    return bboxes