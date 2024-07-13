import torch
import os
import time
import numpy as np
from tqdm.notebook import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils.metrics import mIoU, pixel_accuracy
from data.data import ADE20kSegmentationDataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, unfreeze_layers=[], patch=False, resume_checkpoint=None,device='cpu', epochs_after_unfreeze=19):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_iou = []
    val_iou = []
    train_acc = []
    val_acc = []
    lrs = []
    min_val_loss = np.inf
    decrease = 1
    not_improve = 0
    start_epoch = 0
    
    model.to(device)
    fit_time = time.time()
    
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            min_val_loss = checkpoint['min_val_loss']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            train_iou = checkpoint['train_iou']
            val_iou = checkpoint['val_iou']
            train_acc = checkpoint['train_acc']
            val_acc = checkpoint['val_acc']
            lrs = checkpoint['lrs']
            decrease = checkpoint['decrease']
            not_improve = checkpoint['not_improve']
        else:
            print(f"Checkpoint file not found at {resume_checkpoint}. Starting fresh training.")
    
    for e in range(start_epoch, epochs):
        since = time.time()
        running_loss = 0.0
        running_iou = 0.0
        running_acc = 0.0
        
        # Training phase
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            images, masks = data
            if patch:
                bs, n_tiles, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                masks = masks.view(-1, h, w)
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)['out']
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Evaluation metrics
            running_loss += loss.item()
            running_iou += mIoU(outputs, masks)
            running_acc += pixel_accuracy(outputs, masks)
            
        train_losses.append(running_loss / len(train_loader))
        train_iou.append(running_iou / len(train_loader))
        train_acc.append(running_acc / len(train_loader))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou_score = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                images, masks = data
                if patch:
                    bs, n_tiles, c, h, w = images.size()
                    images = images.view(-1, c, h, w)
                    masks = masks.view(-1, h, w)
                
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                
                val_loss += criterion(outputs, masks).item()
                val_iou_score += mIoU(outputs, masks)
                val_accuracy += pixel_accuracy(outputs, masks)
        
        val_losses.append(val_loss / len(val_loader))
        val_iou.append(val_iou_score / len(val_loader))
        val_acc.append(val_accuracy / len(val_loader))
        
        # Print epoch statistics
        print(f"Epoch [{e+1}/{epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Train mIoU: {train_iou[-1]:.4f}, "
              f"Val mIoU: {val_iou[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, "
              f"Val Acc: {val_acc[-1]:.4f}, "
              f"Time: {(time.time() - since) / 60:.2f} min")
        
        # Adjust learning rate and scheduler step
        lrs.append(get_lr(optimizer))
        scheduler.step()
        
        # Gradually unfreeze layers after 5 epochs
        if e == epochs_after_unfreeze:  # Unfreeze layers after 5 epochs
            print("Unfreezing layers...")
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        # Model saving based on validation loss improvement
        if val_losses[-1] < min_val_loss:
            print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_losses[-1]:.6f}). Saving model...")
            min_val_loss = val_losses[-1]
            decrease += 1
            if decrease % 5 == 0:
                torch.save({
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_val_loss': min_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_iou': train_iou,
                    'val_iou': val_iou,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'lrs': lrs,
                    'decrease': decrease,
                    'not_improve': not_improve
                }, f"checkpoint.pth")
        else:
            not_improve += 1
            if not_improve == 5:
                print("Validation loss did not improve for 5 epochs. Stopping training.")
                break
    
    print(f"Total training time: {(time.time() - fit_time) / 60:.2f} min")
    
    # Return history for analysis or plotting
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_miou': train_iou,
        'val_miou': val_iou,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lrs': lrs
    }
    
    return history
