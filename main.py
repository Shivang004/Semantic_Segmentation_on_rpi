import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from finetune import fit
from data.data import ADE20kSegmentationDataset
from model.lrassp_mobilenetv3_small import load_lraspp_mobilenet_v3_small
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    
def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    
def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


if __name__ == "__main__":
    # Load data
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': 480}
    train_dataset = ADE20kSegmentationDataset(root='ade20k_scene_parsing_selected', split='training', **data_kwargs)
    val_dataset = ADE20kSegmentationDataset(root='ade20k_scene_parsing_selected', split='validation', **data_kwargs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    max_lr = 1e-3
    epochs = 100
    weight_decay = 1e-4
    epochs_after_unfreeze = 10

    # Load pretrained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "lraspp_mbv3_small.pth"
    num_classes = 16
    print("Device: ", device)
    # Load pretrained model
    model = load_lraspp_mobilenet_v3_small(checkpoint_path, num_classes=num_classes,finetuning=True, device=device)
    model.to(device)
    
    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    print(model.eval())
    # Train the model
    history = fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler,device=device,epochs_after_unfreeze=epochs_after_unfreeze)
    
    # Optionally save the trained model
    torch.save(model.state_dict(), 'finetuned_model.pth')
    plot_loss(history)
    plot_score(history)
    plot_acc(history)
