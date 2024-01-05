import torch
import cv2
import numpy as np


def tiling(image, tile_size):
    """
    Tile an image into fixed sized rois with stride half the size of the tile.

    Args:
        image: An array representing the image to predict on with shape (C, H, W).
        tile_size: The size of the tiles
    
    Returns:
        tiles: A list of tiles with shape (C, tile_size, tile_size).
        image_size: The size of the input image after tiling.
    """
    
    image_size = image.shape
    
    stride  = tile_size//2
    image = image.transpose(1, 2, 0)
    if image_size[1] % tile_size != 0:
        image = np.pad(image, ((0, tile_size - image_size[1] % tile_size), (0, 0), (0, 0)), 'constant')
    if image_size[2] % tile_size != 0:
        image = np.pad(image, ((0, 0), (0, tile_size - image_size[2] % tile_size), (0, 0)), 'constant')

    image = image.transpose(2, 0, 1)
    new_img_size = image.shape
    tiles = []
    for y in range(0, new_img_size[1], stride):
        for x in range(0, new_img_size[2], stride):
            if y + tile_size > new_img_size[1] or x + tile_size > new_img_size[2]:
                break
            tmp_img = image[:,y:y+tile_size, x:x+tile_size ]
            tmp_img = np.float32(tmp_img)
            tiles.append(tmp_img)

    return tiles, new_img_size


def run_prediction(image, model, device, tile_size, output_path):
    """
    Runs predictions on a single image.
    This function first rescales the image to [0;1] range, changes the image shape to (C, H, W) and 
    then tiles it into fixed sized tiles with stride half the size of the tile.
    Then inference is performed on each tile and the predictions are combined into a single mask.
    A threshold of 0.5 is used to binarize the predictions.
    Finally, the predictions are saved to the output path.
    
    Args:
        image: An array representing the image to predict on with shape (H, W, C=3).
        model: A PyTorch model to predict with.
        device: The device (cpu or cuda) to run the model on.
        tile_size: The size of the tiles to use for prediction.
        output_path: The path to save the predicted mask to.
    
    Returns:
        None
    """
    image = np.array(image)

    size = image.shape
    image = image / 255.0
    image = image.transpose(2, 0, 1)

    image_tiles, image_size = tiling(image, tile_size)
    mask = np.zeros((image_size[1], image_size[2]), dtype=np.float32)
    
    model.to(device)
    model.eval()

    stride = tile_size // 2

    x = 0
    y = 0
    for tile in image_tiles:
        img = np.expand_dims(tile, axis=0)
        img = torch.tensor(img)
        img = img.to(device)

        with torch.no_grad():
            mask_pred = model(img)
        mask_pred = mask_pred.squeeze(0).squeeze(0).cpu().detach().numpy()
        cv2.threshold(mask_pred, 0.5, 1, cv2.THRESH_BINARY, mask_pred)
        
        mask[y:y+tile_size, x:x+tile_size] = np.ma.mask_or(mask[ y:y+tile_size, x:x+tile_size], mask_pred )

        x += stride
        if x + tile_size > mask.shape[1]:
            x = 0
            y += stride
        if y + tile_size > mask.shape[0]:
            break
    mask = mask[:size[0], :size[1]]

    cv2.imwrite(f"{output_path}_mask.png", mask * 255)