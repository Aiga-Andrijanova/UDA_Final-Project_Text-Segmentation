import torch
import os
import cv2

from pdf2image import convert_from_path
from seg_helpers import run_prediction
from train import load_checkpoint
from model import Unet3Plus


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_size = 512
    model_path = "./results/Unet3+_restart_2024-01-02_16-15-36/image_model_best_iou-0.761.pth"

    input_dirs = ["./data/test_data/digital_copies", "./data/test_data/photos"]  # A folder that contains PDFs, PNGs or JPGs
    output_dir = "./inference_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = Unet3Plus()
    model, _, _ = load_checkpoint(model, None, model_path)

    for input_dir in input_dirs:
        for filename in os.scandir(input_dir):
            if filename.is_file() and '.pdf' in filename.name:
                images = convert_from_path(filename.path)
                for idx, img in enumerate(images):
                    output_path = f"{output_dir}/{filename.name[:-4]}_{idx}"
                    run_prediction(img, model, device, tile_size, output_path)
            elif filename.is_file() and ('.jpg' in filename.name or '.png' in filename.name):
                img = cv2.imread(filename.path)
                output_path = f"{output_dir}/{filename.name[:-4]}"
                run_prediction(img, model, device, tile_size, output_path)
            print(f"Processed: {filename.name}")

    print(f"\n Inference finished!")
