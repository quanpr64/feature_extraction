from openslide import open_slide
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage import io
from skimage.color import rgb2hsv
from skimage.transform import resize
from skimage.filters import threshold_otsu
import os, argparse
from math import ceil
from tempfile import TemporaryDirectory
from traceback import print_exc
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel, ViTModel
import timm
import uuid

from torchvision import transforms

# Set HuggingFace token
os.environ["HF_TOKEN"] = "hf_*****"   # add your hugging face token to run this script
assert "HF_TOKEN" in os.environ


def load_extractor(model_name):
    """
    Load pretrained feature extractor based on model name.
    """
    if model_name == 'phikon':
        extractor = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    elif model_name == 'phikon2':
        extractor = AutoModel.from_pretrained("owkin/phikon-v2")
    elif model_name == 'prov-gigapath':
        extractor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name == 'uni':
        extractor = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, patch_size=16, num_classes=0, img_size=224, dynamic_img_size=True)
    elif model_name == 'uni2-h':
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        extractor = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return extractor




def _compute_tissue_mask_from_array(arr):
    """
    Compute binary tissue mask from an RGB image using HSV color space and Otsu thresholding.

    Args:
        arr (np.ndarray): RGB image array (H, W, 3)

    Returns:
        mask (np.ndarray): Binary mask (H, W), tissue area as 1, background as 0
    """
    hsv = rgb2hsv(arr)
    threshold_h = threshold_otsu(hsv[:, :, 0])
    threshold_s = threshold_otsu(hsv[:, :, 1])
    tissue_mask = np.logical_and(hsv[:, :, 0] > threshold_h,
                                 hsv[:, :, 1] > threshold_s)
    kernel = np.ones((1, 1), dtype=np.uint8)
    mask = cv2.dilate(tissue_mask.astype(np.uint8), kernel, iterations=1)
    return mask

def compute_tissue_mask(slide):
    """
    Estimate tissue mask from WSI thumbnail.

    Args:
        slide (openslide.OpenSlide): Opened WSI object

    Returns:
        mask_resized (np.ndarray): Tissue mask with same shape as level 0 (uint8, 0/1)
        base_shape (tuple): Original image shape (H, W)
    """
    base_w, base_h = slide.level_dimensions[0]
    base_shape = (base_h, base_w)
    thumb_size = (base_w // 16, base_h // 16)
    thumb_img = slide.get_thumbnail(thumb_size)
    thumbnail_arr = np.array(thumb_img)[..., :3]
    mask_small = _compute_tissue_mask_from_array(thumbnail_arr)
    mask_resized = resize(
        mask_small,
        base_shape,
        order=0,
        preserve_range=True
    ).astype(np.uint8)
    return mask_resized, base_shape

def _compute_tiles_coordinates_from_mask(mask,
                                         fullres_w,
                                         fullres_h,
                                         tile_size=224,
                                         matter_threshold=0.6,
                                         seed=None):
    """
    Compute candidate tile coordinates from a tissue mask.

    Args:
        mask (np.ndarray): Tissue mask with values 0 or 1
        fullres_w (int): Full resolution image width
        fullres_h (int): Full resolution image height
        tile_size (int): Tile size in pixels
        matter_threshold (float): Minimum tissue ratio per tile
        seed (int): Random seed for shuffling

    Returns:
        coords (list of tuple): Top-left (x, y) coordinates of valid tiles
    """
    mask_pil = Image.fromarray((mask * 255).astype("uint8"))
    num_tiles_w = ceil(fullres_w / tile_size)
    num_tiles_h = ceil(fullres_h / tile_size)
    resized_mask = mask_pil.resize((num_tiles_w, num_tiles_h), resample=Image.NEAREST)
    mask_array = np.array(resized_mask)
    coords = list(zip(*np.where(mask_array.T > matter_threshold * 255)))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(coords)
    return coords

def from_tiles_coords_to_pixels_coords(tiles_coords,
                                       fullres_w,
                                       fullres_h,
                                       tile_size=224):
    """
    Convert tile grid coordinates to pixel coordinates.

    Args:
        tiles_coords (list of tuple): Tile grid coordinates
        fullres_w (int): Full resolution width
        fullres_h (int): Full resolution height
        tile_size (int): Tile size

    Returns:
        px_coords (list of tuple): Pixel-level top-left coordinates
    """
    _px_coords = [(int(x * tile_size), int(y * tile_size)) for x, y in tiles_coords]
    px_coords = [tup for tup in _px_coords if tup[0] + tile_size < fullres_w and tup[1] + tile_size < fullres_h]
    return px_coords

def get_tiles_coordinates(slide,
                          tile_size=224,
                          matter_threshold=0.6,
                          seed=None):
    """
    Compute pixel coordinates of tissue-containing tiles.

    Args:
        slide (openslide.OpenSlide): WSI
        tile_size (int): Tile size
        matter_threshold (float): Tissue threshold
        seed (int): Random seed

    Returns:
        px_coords (list of tuple): Valid pixel coordinates
    """
    mask, fullres_shape = compute_tissue_mask(slide)
    fullres_w, fullres_h = fullres_shape[1], fullres_shape[0]
    tiles_coords = _compute_tiles_coordinates_from_mask(
        mask=mask,
        fullres_w=fullres_w,
        fullres_h=fullres_h,
        tile_size=tile_size,
        matter_threshold=matter_threshold,
        seed=seed)
    px_coords = from_tiles_coords_to_pixels_coords(tiles_coords, fullres_w, fullres_h, tile_size)
    return px_coords

class ImageDataset(Dataset):
    """
    PyTorch dataset for loading image tiles from a directory.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.__build()

    def __build(self):
        self._paths = list(Path(self.root_dir).rglob('*.tiff'))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        img = Image.open(self._paths[idx])
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

def tiling_with_split_png(slide,
                          output_dir,
                          px_coords,
                          max_tiles_per_slide,
                          tile_size):
    """
    Extract and save tiles from slide based on pixel coordinates.
    """
    print("num of px_coords: {}".format(len(px_coords)))
    if max_tiles_per_slide:
        px_coords = px_coords[:max_tiles_per_slide]
    for coord in px_coords:
        region = slide.read_region((coord[0], coord[1]), level=0, size=(tile_size, tile_size))
        img = np.array(region)[..., :3]
        io.imsave(os.path.join(output_dir, f"{coord[0]}_{coord[1]}.tiff"), img, check_contrast=False)

def save_coords_to_file(coords, tile_size, max_tiles_per_slide, file_path):
    """
    Save pixel coordinates of tiles to CSV file.
    """
    with open(file_path, mode='w') as file:
        file.write(f"x0,y0,x1,y1\n")
        if max_tiles_per_slide:
            coords = coords[:max_tiles_per_slide]
        for coord in coords:
            file.write(f"{coord[0]},{coord[1]},{coord[0]+tile_size},{coord[1]+tile_size}\n")

def _inference(model, loader, device, model_name):
    """
    Run batched inference with given model and dataloader.
    """
    features = []
    with torch.inference_mode():
        for images in loader:
            batch_size = images.shape[0]
            images = images.to(device, non_blocking=True)
            output = model(images)
            if 'phikon' in model_name:
                batch_features = output.last_hidden_state[:, 0, :]
            elif 'uni' in model_name:
                batch_features = output
            elif 'gigapath' in model_name:
                batch_features = output
            batch_features = batch_features.detach().cpu().numpy()
            features.append(batch_features)
    return np.concatenate(features, axis=0)

def adjust_tile_size_for_mpp(mpp, target_tile_size=224):
    """
    Adjust tile size according to microns per pixel.
    """

    REF_MPP = 0.5 #
    adjusted_size = target_tile_size * (REF_MPP / mpp)
    return int(round(adjusted_size))


# 5. extract

def _extract_from_single_slide(slide_path,
                               extractor,
                               model_name,
                               tile_size = 224,
                               matter_threshold = 0.6,
                               batch_size = 16,
                               num_workers = 4,
                               max_tiles = None,
                               seed = 1,
                               coords_save_path = None,
                               temp_dir = None,
                               input_mpp = 0.5,
                               device = 'cuda:0'):
    
    """Runs feature extraction for a SINGLE slide."""
    if not temp_dir:
        temp_dir = os.path.join(os.getcwd(), str(uuid.uuid1()))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    with TemporaryDirectory(dir=temp_dir) as tmp_dir:
        # content in TemporaryDirectory will delete after finish
        try:
            print(slide_path)
            slide = open_slide(slide_path)
            mpp_x = float(slide.properties.get("openslide.mpp_x", input_mpp))
            mpp_y = float(slide.properties.get("openslide.mpp_y", input_mpp))
            mpp = (mpp_x+mpp_y)/2
            target_tile_size = adjust_tile_size_for_mpp(mpp, tile_size)
            
            # Step 1: coordinates of tiles with matter
            px_coords = get_tiles_coordinates(slide,
                                                 tile_size=target_tile_size,
                                                 matter_threshold=matter_threshold,
                                                 seed=seed)
            
            # Step 2: save coordinates and pathes
            if coords_save_path:
                file_path = os.path.join(coords_save_path, 'px_coords_' + slide_path.split('/')[-1] + '.csv')
                save_coords_to_file(px_coords, target_tile_size, max_tiles, file_path)
            
            tiling_with_split_png(slide,
                          output_dir=tmp_dir,
                          px_coords=px_coords,
                          max_tiles_per_slide=max_tiles,
                          tile_size=target_tile_size)
            
            # Step 3: feature extraction

            transform_ops = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]

            if target_tile_size != tile_size:
                transform_ops_ = [transforms.Resize((tile_size, tile_size), Image.BICUBIC)] + transform_ops
                _transform = transforms.Compose(transform_ops_)
            else:
                _transform = transforms.Compose(transform_ops)
            
            dataset = ImageDataset(root_dir=tmp_dir, transform=_transform)
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=False,
                                pin_memory=True)
            features = _inference(extractor, loader, device, model_name)

            # Clear
            del dataset

            return features

        except Exception: 

            print(f'--- Preprocessing of slide {slide_path} failed with:')
            print_exc()

             
def main(args):
    if not os.path.exists(args.feature_save_path):
        os.makedirs(args.feature_save_path)
    if args.coords_save_path and not os.path.exists(args.coords_save_path):
        os.makedirs(args.coords_save_path)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Cuda is not available, use cpu instead.")
        args.device = 'cpu'
    device = torch.device(args.device)



    extractor = load_extractor(args.model_name)
    extractor.eval()
    extractor.to(device)
    
    df = pd.read_csv(args.input_csv)
    
    for i in range(len(df)):
        if 'features_'+df['image_path'][i].split('/')[-1]+'.npy' in os.listdir(args.feature_save_path):
            print('features_'+df['image_path'][i]+'.npy exists!!!')
            continue
        f = _extract_from_single_slide(df['image_path'][i],
                                       extractor,
                                       model_name = args.model_name,
                                       tile_size = args.tile_size,
                                       matter_threshold = args.matter_threshold,
                                       batch_size = args.batch_size,
                                       num_workers = args.num_workers,
                                       max_tiles = args.max_tiles,
                                       seed = args.seed,
                                       coords_save_path = args.coords_save_path,
                                       temp_dir = args.temp_dir,
                                       input_mpp = args.input_mpp,
                                       device = device)
        np.save(os.path.join(args.feature_save_path, 'features_' + str(df['image_path'][i].split('/')[-1]) + '.npy'), f)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from slides')

    parser.add_argument('--input_csv', type=str,
                        help="image info csv file includes column 'image_path'", required=True)
    parser.add_argument('--feature_save_path', type=str, 
                    help='output folder for features per slide', required=True)
    parser.add_argument('--coords_save_path', default=None, type=str, 
                    help='output folder for patch coords per slide')
    
    parser.add_argument('--batch_size', default=256, type=int, 
                    help='batch size')
    parser.add_argument('--num_workers', default=64, type=int, 
                    help='number of workers')
    parser.add_argument('--tile_size', default=224, type=int, 
                    help='number of workers')
    parser.add_argument('--device', default='cpu',  type=str, help='Run on cpu or gpu.')
    parser.add_argument('--input_mpp', default=0.5, type=float, help='Input data mpp.')
    
    parser.add_argument('--matter_threshold', default=0.6, type=float,
                        help='A patch includes at least matter_threshold * 100% of tissue area.')
    parser.add_argument('--seed', default=100000, type=int, help='Random seed.')
    parser.add_argument('--max_tiles', default=None, type=int, 
                    help='number of tiles to extract')
    parser.add_argument('--temp_dir', default=None, type=str, 
                    help='temp dir for saving patches, will delete once finish')
    
    parser.add_argument('--model_name', default='phikon', choices=['phikon', 'uni', 'phikon2', 'prov-gigapath', 'uni2-h'],
                        type=str, help="Model name")
    
    args = parser.parse_args()
    main(args)



