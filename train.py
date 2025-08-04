# train.py

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from dataset import IndianNumberPlateDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = IndianNumberPlateDataset(
        images_dir='data/indian_plates/images',
        annotations_dir='data/indian_plates/annotations',
        transforms=ToTensor()
    )

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes=2)  # background + number plate
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "fasterrcnn_numberplate.pt")
    print("✅ Model saved as fasterrcnn_numberplate.pt")

if __name__ == "__main__":
    train()
