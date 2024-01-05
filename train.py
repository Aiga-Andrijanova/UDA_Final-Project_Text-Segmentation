import torch
import os
import cv2
import time
import datetime
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

from torchmetrics import JaccardIndex
from pdf2image import convert_from_path
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from seg_helpers import run_prediction
from dataset import *
from pathlib import Path
from model import Unet3Plus, DiceLoss

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-checkpoint_path', 
                    default="./results/Unet3+_restart_2024-01-02_16-15-36/image_model_best_iou-0.761.pth", type=str,
                    help="Path to .pth file where model weights are saved")
parser.add_argument('-project_dir', default="./results", type=str,
                    help="Path to directory where to save run results")
parser.add_argument('-run_name', 
                    default='Unet3+_restart', type=str,
                    help="Run name")

# Training parameters
parser.add_argument('-epoch_count', default=5_000, type=int)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-checkpoint_interval', default=1, type=int,
                    help="Interval after how many epochs save model state")
parser.add_argument('-run_inference', default=True, type=bool)
parser.add_argument('-inference_interval', default=5, type=int,
                    help="If run_inference set to true, interval after how many epochs perform inference")
parser.add_argument('-imgs_to_save', default=4, type=int,
                    help="How many images save in tensorboard per train and evaluation sets")

# Data params
parser.add_argument('-data_paths', default=["/home/aiga/Projects/ocr-segmentation/ocr-generator/generated/train_documents_with_backgrounds_tiled", 
                                            "/home/aiga/Projects/ocr-segmentation/ocr-generator/generated/extracted_eis_with_backgrounds_tiled"],
                                            help="A list to root folders of the generated and tiled samples")
parser.add_argument('-test_data_path', default="./results/test_data/photos", type=str,
                    help="A list to a folder containing test data (.png, .jpg, .jpeg or .pdf files)")
parser.add_argument('-percentage_of_data_to_use', default=1.0, type=float,
                    help="How much of the samples to use, debug argument")
parser.add_argument('-train_split_size', default=0.9, type=float,
                    help="How much of the documents put in the train set")
parser.add_argument('-num_workers', default=12 , type=int,
                    help="How many dataloader workers spawn, if set to 0, none will spawn")
parser.add_argument('-batch_prefetch_factor', default=2, type=int,
                    help="Batch prefetch fatctor for the dataloader")
parser.add_argument('-input_shape', default=[512, 512, 3], nargs='+', type=int)

args, other_args = parser.parse_known_args()


def load_checkpoint(model, optimizer, filename):
    """
    Load model and optimizer state from a checkpoint.
    
    Args:
        model: A PyTorch model to load the weights to.
        optimizer: A PyTorch optimizer to load the state to.
        filename: The path to the checkpoint file to load.
    
    Returns:
        model: The model with loaded weights.
        optimizer: The optimizer with loaded state.
        start_epoch: The epoch to start training from.
    """
    start_epoch = 0
    if os.path.isfile(filename):
        try:
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("📂 Loaded checkpoint '{}' (epoch {}) \n"
                    .format(filename, checkpoint['epoch']))
        except KeyError:
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint)
            print("📂 Loaded checkpoint '{}' without optimizer and start epoch \n"
                    .format(filename))
    else:
        print("🚧 No checkpoint found at '{}'\n".format(filename))

    return model, optimizer, start_epoch


def save_model(checkpoint_dir, model, optimizer, epoch, checkpoint_name):
    """
    Save model and optimizer state to a checkpoint.
    
    Args:
        checkpoint_dir: The directory to save the checkpoint to.
        model: A PyTorch model to save the weights from.
        optimizer: A PyTorch optimizer to save the state from.
        epoch: The current epoch to save.
        checkpoint_name: The name to save the checkpoint under.
    
    Returns:
        None
    """
    checkpoint_path = checkpoint_dir/f"{checkpoint_name}.pth"
    state = {'epoch': epoch + 1, 
             'model_state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)


def train_loop(model, train_dataloader, val_dataloader, optimizer,device, start_epoch, epochs, checkpoint_dir, tensorboard_writer):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        val_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        device: The device (cpu or cuda) to run the model on.
        start_epoch: The epoch training starts from
        epochs: The number of epochs to train for.
        checkpoint_dir: The dir where to save the best model checkpoints and tensorbaord files.
    
    Returns:
        None
    """
    dice_loss = DiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    jaccard = JaccardIndex(task="binary", num_classes=1)

    best_iou = 0
    for epoch in tqdm(range(start_epoch, epochs, 1), desc="Epochs"):
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
                data_loader = train_dataloader
            else:
                model.eval() 
                data_loader = val_dataloader
          
            epoch_loss = 0
            epoch_iou = 0
           
            start_time = time.time()
            for idx, batch in enumerate(tqdm(data_loader)):
                imgs, masks, names = batch
                images = imgs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
            
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)
                    masks_pred = model(images)
                else:
                    with torch.no_grad():
                        masks_pred = model(images)

                loss = criterion(masks_pred.squeeze(1), masks)
                loss += dice_loss(masks_pred.squeeze(1), masks)
                
                iou = jaccard(masks_pred.squeeze(1).cpu(), masks.cpu())

                if idx == len(data_loader) - 2 and epoch%args.checkpoint_interval==0: # Save images on the batch before the last one so that there are at least 4 images in the batch
                    img_batch = torch.cat((images[:args.imgs_to_save,...], masks[:args.imgs_to_save,...].unsqueeze(1).expand(-1, 3, -1, -1), masks_pred[:args.imgs_to_save,...].expand(-1, 3, -1, -1)), 2)
                    tensorboard_writer.add_images(f'Input images, ground truths and predicted masks/{mode}', img_batch[:args.imgs_to_save,...], epoch)
                    name_str ="".join([f"  \n{names[i]}" for i in range(args.imgs_to_save)])
                    tensorboard_writer.add_text(f"Metadata/{mode}", f'loss: {loss}  \niou: {iou}' + name_str, epoch)
                
                tensorboard_writer.add_scalar(f"Batch_loss/{mode}", loss, epoch * len(data_loader) + idx)
                tensorboard_writer.add_scalar(f"Batch_IOU/{mode}", iou, epoch * len(data_loader) + idx)
                
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item()
                epoch_iou += iou.item()

            end_time = time.time()
            epoch_loss = epoch_loss / len(data_loader)
            epoch_iou = epoch_iou / len(data_loader)
            
            tensorboard_writer.add_scalar(f"Epoch_loss/{mode}", epoch_loss , epoch)
            tensorboard_writer.add_scalar(f"Epoch_IOU/{mode}", epoch_iou, epoch)

            print(
                'Mode: {}, Epoch: {}, Time: {:.6f} s, Loss: {:.6f}, IoU: {:.6f}'.format(
                    mode, epoch, end_time-start_time, epoch_loss, epoch_iou
                )
            )
            
            if mode=='val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                save_model(checkpoint_dir, model, optimizer, epoch, f"image_model_best_iou-{round(epoch_iou, 3)}")

            if mode == 'val' and epoch%args.checkpoint_interval==0:
                save_model(checkpoint_dir, model, optimizer, epoch, f"image_model_epoch-{epoch}_iou-{round(epoch_iou, 3)}")

            if mode == 'val' and args.run_inference is True and epoch%args.inference_interval==0:
                output_dir = f"{checkpoint_dir}/inference_result_epoch-{epoch}_iou-{epoch_iou}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                input_dir = args.test_data_path # A folder that contains PDFs and JPGs
                for filename in os.scandir(input_dir):
                    if filename.is_file() and '.pdf' in filename.name:
                        images = convert_from_path(filename.path)
                        for idx, img in enumerate(images):
                            output_path = f"{output_dir}/{filename.name[:-4]}_{idx}"
                            run_prediction(img, model, device, args.input_shape[0], output_path)
                    elif filename.is_file() and ('.jpg' in filename.name or '.jpeg' in filename.name or '.png' in filename.name):
                        img = cv2.imread(filename.path)
                        output_path = f"{output_dir}/{filename.name[:-4]}"
                        run_prediction(img, model, device, args.input_shape[0], output_path)
                print(f"Inference completed")
            
            if device.type != 'cpu': # If the device is a GPU, empty the cache
                getattr(torch, device.type).empty_cache()
                
        writer.flush()


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_dir = Path(args.project_dir)
    checkpoint_dir = Path(project_dir/f"{args.run_name}_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Training is logged in {checkpoint_dir} \n")

    train_transform = A.Compose([
                A.RandomResizedCrop(height=args.input_shape[0], width=args.input_shape[0], scale=(0.20,1.0), ratio=(0.8, 1.2), p=1.0),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.25),
                A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05, p=0.75),
                A.RandomGamma(p=0.50),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 70), p=0.2), 
                    A.ImageCompression(quality_lower=97, p=0.2),
                    A.ISONoise(intensity=(0.1, 0.5), p=0.2),
                    A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.2),
                    A.AdvancedBlur(blur_limit=(3, 5), p=0.2),
                    A.Blur(blur_limit=(3, 5), p=0.2)],
                      p=0.5),
                ToTensorV2()])
        
    val_transform = A.Compose([
            A.RandomResizedCrop(height=args.input_shape[0], width=args.input_shape[0], scale=(0.20,1.0), ratio=(0.8, 1.2), p=1.0),
            ToTensorV2()])
    
    train_items, val_items = dataset_splitter(args.data_paths, args.train_split_size, args.percentage_of_data_to_use, seed=0)

    train_dataset = PaperworkDataset(train_items, args.input_shape, train_transform, args.input_shape[0])
    val_dataset = PaperworkDataset(val_items, args.input_shape, val_transform, args.input_shape[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=args.num_workers,
                                               prefetch_factor=args.batch_prefetch_factor)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             prefetch_factor=args.batch_prefetch_factor)

    print(f"📁 Train dataset len: {len(train_dataset)}, train loader batch count: {len(train_loader)}")
    print(f"📂 Val dataset len: {len(val_dataset)}, val loader batch count: {len(val_loader)} \n")

    writer = SummaryWriter(log_dir=checkpoint_dir)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🤖 Using computational power: {device} \n")

    model = Unet3Plus()
    model.to(device)

    optimizer = torch.optim.RAdam(params=model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    epochs=args.epoch_count
    if len(args.checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        epochs += start_epoch

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📐 Model initialized \n"
          f"🧮 All parameter count: {total_params}, \n"
          f"📏 Trainable parameter count: {trainable_params} \n")

    print(f"⌛ Model training will start from epoch no {start_epoch} and run till epoch no {epochs}  \n")

    train_loop(model=model, 
           train_dataloader=train_loader,
           val_dataloader=val_loader,
           optimizer=optimizer,
           device=device, 
           start_epoch = start_epoch,
           epochs=epochs, 
           checkpoint_dir=checkpoint_dir,
           tensorboard_writer = writer)

    writer.close()