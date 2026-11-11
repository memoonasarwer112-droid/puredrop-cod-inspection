import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PureDropDataset
from model import BaselineCODModel

def train_model():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = PureDropDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=transform)
    
    if len(train_dataset) == 0:
        print("Warning: No training images found in 'data/train/images'! Please add some image/mask pairs first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCODModel().to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training started on device: {device}")

    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/1], Loss: {running_loss/len(train_loader):.4f}")

    print("Training script executed successfully!")

if __name__ == "__main__":
    train_model()