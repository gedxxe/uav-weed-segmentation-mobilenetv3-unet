from pathlib import Path
from skimage import io as skio
import numpy as np
import math

from utils.labels import LABEL_COLORS


def get_idx_to_remove(anns):
    '''
    get the indices of patches with only background or really small plants (sum is less than 1% of the total pixels)
    Note: weed pixels are counting x2
    '''
    remove_ids = []
    for idx, ann in enumerate(anns):
        if ann.reshape(-1).sum() <=655:
            remove_ids.append(idx)
    return remove_ids


def get_file_lists(data_path, subset):
    path = data_path / subset
    imgPaths = list(path.glob('./img/*.jpg'))
    img_list = sorted(imgPaths)
    annPaths = list(path.glob('./msk/*.png'))
    msk_list = sorted(annPaths)

    if len(img_list) == 0 or len(msk_list) == 0:
        raise FileNotFoundError(f"No raw images/masks found in {path}. Expected img/*.jpg and msk/*.png.")
    if len(msk_list) != len(img_list):
        raise ValueError(f"Different amount of images and masks in {path}: {len(img_list)} images, {len(msk_list)} masks")
    print(f"Found {len(msk_list)} images and masks in {str(path)}")
    return img_list, msk_list 

def load_images(img_ls):
    """
    img_ls: list of image paths to laod.
    returns: list of patches of images
    """
    
    imgs = []
    for imgPath in img_ls:
        img = load_image(path = imgPath)
        patches, slc_size = image_to_patches(image=img, b_msk=False)
        imgs.append(patches)
    imgs = np.concatenate(imgs, axis=0)
    return imgs

def load_image(path):
    """
    loads an image based on the path
    """
    rgb_image = skio.imread(path)
    return rgb_image

def load_masks(msk_ls, labels=LABEL_COLORS):
    """
    msk_ls: list of mask paths to load. should be in the same order as the images
    returns: list of patches of masks
    """
    anns = []
    for annPath in msk_ls:
        ann = load_mask(path = annPath, labels=labels)
        patches, slc_size = image_to_patches(image=ann, b_msk=True)
        anns.append(patches)
    anns = np.concatenate(anns, axis=0)
    return anns

def load_mask(path, labels=LABEL_COLORS):
    """
    loads a mask based on the path and encodes it according to a list of RGB tuples 
    """
    rgb_mask = skio.imread(path)
    label_map = set_label_map(rgb_mask, labels)
    return label_map

def set_label_map(rgb_mask, labels):
    """
    encodes 3D RGB Mask into 2D array based on a List of RGB tuples
    """
    label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for idx, label in enumerate(labels):
        label_map[(rgb_mask==label).all(axis=2)] = idx
    
    return label_map

def image_to_patches(image, slc_size=256, b_msk=False):
    x = int(math.ceil(image.shape[0]/(slc_size * 1.0)))
    y = int(math.ceil(image.shape[1]/(slc_size * 1.0)))
    padded_shape = (x*slc_size, y*slc_size)
    if b_msk==False:
        padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1], 3), dtype=np.uint8)
        padded_rgb_image[:image.shape[0],:image.shape[1]] = image
        patches = padded_rgb_image.reshape(x, slc_size, y, slc_size, 3)
        patches = patches.swapaxes(1, 2).reshape(-1, slc_size, slc_size, 3)
    elif b_msk==True:
        padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1]), dtype=np.uint8)
        padded_rgb_image[:image.shape[0],:image.shape[1]] = image
        patches = padded_rgb_image.reshape(x, slc_size, y, slc_size)
        patches = patches.swapaxes(1, 2).reshape(-1, slc_size, slc_size)

    return patches, slc_size

def save_patches(patches, path_to_save, subset, postfix):
    for idx, val in enumerate(range(patches.shape[0])):
        skio.imsave(f"{path_to_save}/{subset}_p_{val:05d}_{postfix}.png", patches[idx], check_contrast=False)
