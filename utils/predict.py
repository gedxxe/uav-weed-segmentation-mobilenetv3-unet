import numpy as np
import torch
import os
import math
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.dataset import UAVDatasetPatches
from utils.labels import LABEL_COLORS
from utils.model_outputs import get_segmentation_logits
from utils.train import autocast_context

def get_test_loader(test_img_dir, test_msk_dir, mean, std, batch_size, num_workers=4, pin_memory=True):

    test_transform = A.Compose(
        [
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    test_ds = UAVDatasetPatches(img_list=test_img_dir, msk_list=test_msk_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return test_loader

def predict(model, test_loader, device, use_amp=True):
    '''
    predicts all images in the test_loader
    '''
    model.eval()
    predictions_whole = None 

    for inputs, targets in test_loader:
        with torch.no_grad():
            predictions = predict_one_batch(model, inputs, targets, device, use_amp=use_amp)
            if predictions_whole is None:
                predictions_whole = predictions
            else:
                predictions_whole = torch.cat((predictions_whole, predictions), dim=0)
    return predictions_whole

def predict_one_batch(model, inputs, targets, device, use_amp=True):
    '''
    validates one batch
    '''
    with autocast_context(device, use_amp):
        inputs = inputs.float().to(device=device)
        targets = targets.long().to(device=device)

        predictions = get_segmentation_logits(model(inputs))
        predicted_masks = torch.argmax(predictions, dim=1)
    return predicted_masks

def convert_labelmap_to_color(labelmap, labels=LABEL_COLORS):
    '''
    Colors the 1 channel output into a RGB Image
    '''   
    lookup_table = np.array(labels)
    result = np.zeros((*labelmap.shape,3), dtype=np.uint8)
    np.take(lookup_table, labelmap, axis=0, out=result)
    return result

def combine_labelmap_from_slices(labelmap, grid, slc_size=256):
    '''
    input: torch tensor in gpu with shape NxWxH or NxCxWxH
    takes a labelmap of the shape of BxWxH and converts it to WxH, corresponding a whole capture
    '''
    if len(labelmap.shape) == 3:
        labelmap = labelmap.cpu().numpy()
        full_ann = np.zeros((grid[1]*slc_size, grid[0]*slc_size),dtype=np.uint8)
        offset = (slc_size,slc_size)
        tile_size= (slc_size,slc_size)
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1])] = labelmap[placement]
                placement+=1
    elif len(labelmap.shape) ==4:
        # reshape 
        labelmap = labelmap.permute(0,2,3,1).cpu().numpy()
        
        full_ann = np.zeros((grid[1]*slc_size, grid[0]*slc_size, 3))
        offset = (slc_size,slc_size)
        tile_size= (slc_size,slc_size)
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1]),:] = labelmap[placement]
                placement+=1
    return full_ann


def grid_from_mask_shape(mask_shape, slc_size=256):
    height, width = mask_shape[:2]
    grid_cols = int(math.ceil(width / float(slc_size)))
    grid_rows = int(math.ceil(height / float(slc_size)))
    return grid_cols, grid_rows

def get_slices_per_image(labelmap, slc_per_image):
    '''
    returns a list with the length of images, with the labelmap_per_image as BxWxH as each item
    '''
    if labelmap.shape[0] % slc_per_image != 0:
        raise ValueError(
            f"Prediction slice count {labelmap.shape[0]} is not divisible by "
            f"the expected slices per image ({slc_per_image})."
        )
    num_images = int(labelmap.shape[0]/slc_per_image)
    labelmaps =[]
    for i in range(num_images):
        labelmap_per_image = labelmap[i*slc_per_image:(i+1)*slc_per_image,:,:] 
        labelmaps.append(labelmap_per_image)
    return labelmaps

def reshape_predictions_to_images(preds, labels=LABEL_COLORS, mask_shape=None, slc_size=256):
    predictions_color = []
    if mask_shape is None:
        raise ValueError("mask_shape is required so the prediction grid can be derived dynamically.")
    grid = grid_from_mask_shape(mask_shape, slc_size=slc_size)
    slc_per_image = grid[0] * grid[1]

    preds_labelmaps = get_slices_per_image(preds, slc_per_image=slc_per_image)
    for lab in preds_labelmaps:
        lab_full = combine_labelmap_from_slices(lab, grid=grid, slc_size=slc_size)
        lab_full = lab_full[0:mask_shape[0], 0:mask_shape[1]]
        prediction = convert_labelmap_to_color(lab_full, labels=labels)
        predictions_color.append(prediction)       
    return predictions_color
