# import os
# import random
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from tensorflow.keras import backend as K

# plt.style.use("ggplot")


# def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
#     fig = plt.figure(figsize=(12, 12))
#     for i in range(1, rows*columns+1):
#         fig.add_subplot(rows, columns, i);
#         img_path = list_img_path[i]
#         mask_path = list_mask_path[i]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path)
#         plt.imshow(image)
#         plt.imshow(mask, alpha=0.4)
#     plt.show()

# # 2*area of overlab divided by the total number of pixels in both images
# def dice_coefficients(y_true, y_pred, smooth = 100):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)

#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
#     return (2*intersection + smooth)/ (union + smooth)

# def dice_coefficients_loss(y_true, y_pred, smooth = 100):
#     return -dice_coefficients(y_true, y_pred, smooth)

# def iou(y_true, y_pred, smooth = 100):
#     intersection = K.sum(y_true * y_pred)
#     sum = K.sum(y_true + y_pred)
#     iou = (intersection + smooth) / (sum - intersection + smooth)
#     return iou

# def jaccard_distance(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)

#     return -iou(y_true_flatten, y_pred_flatten)
        
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.style.use("ggplot")

def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    """
    Plots a grid of images with corresponding masks overlaid.
    `list_img_path` and `list_mask_path` should be lists of file paths.
    """
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)

        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)
    plt.show()

import torch

def dice_coefficients(y_true: torch.Tensor, 
                      y_pred: torch.Tensor, 
                      smooth: float = 100) -> torch.Tensor:
    """
    Computes the Dice coefficient: 2 * (|X ∩ Y|) / (|X| + |Y|).
    y_true, y_pred: shape [N, ..., H, W], with values in [0,1].
    smooth: helps avoid division by zero.
    Returns a scalar (averaged across the batch).
    """
    # Flatten each sample in the batch
    y_true_f = y_true.view(y_true.size(0), -1)
    y_pred_f = y_pred.view(y_pred.size(0), -1)

    intersection = (y_true_f * y_pred_f).sum(dim=1)
    union = y_true_f.sum(dim=1) + y_pred_f.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()  # average across batch

def dice_coefficients_loss(y_true: torch.Tensor, 
                           y_pred: torch.Tensor, 
                           smooth: float = 100) -> torch.Tensor:
    """
    Dice loss = 1 - Dice coefficient (or negative dice).
    """
    return 1.0 - dice_coefficients(y_true, y_pred, smooth)

def iou(y_true: torch.Tensor, 
        y_pred: torch.Tensor, 
        smooth: float = 100) -> torch.Tensor:
    """
    Intersection over Union (Jaccard Index):
    IoU = |X ∩ Y| / |X ∪ Y|
    with 'smooth' to avoid zero division.
    Returns average IoU across the batch.
    """
    # Flatten
    y_true_f = y_true.view(y_true.size(0), -1)
    y_pred_f = y_pred.view(y_pred.size(0), -1)

    intersection = (y_true_f * y_pred_f).sum(dim=1)
    union = y_true_f.sum(dim=1) + y_pred_f.sum(dim=1) - intersection
    iou_val = (intersection + smooth) / (union + smooth)
    return iou_val.mean()

def jaccard_distance(y_true: torch.Tensor, 
                     y_pred: torch.Tensor, 
                     smooth: float = 100) -> torch.Tensor:
    """
    Jaccard Distance = 1 - IoU
    or negative IoU if you prefer that as a loss.
    """
    return 1.0 - iou(y_true, y_pred, smooth)

