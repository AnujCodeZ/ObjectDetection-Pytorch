import torch
from PIL import Image

import utils
import transforms as T
from engine import train_one_epoch, evaluate

from model import get_model_instance_segmentation
from data import PennFudanDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

trainset = PennFudanDataset('PennFudanPed', get_transform(train=True))
testset = PennFudanDataset('PennFudanPed', get_transform(train=False))

indeces = torch.randperm(len(trainset)).tolist()
trainset = torch.utils.data.Subset(trainset, indeces[:-50])
testset = torch.utils.data.Subset(testset, indeces[-50:])

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=2,
                                          num_workers=4,
                                          shuffle=True,
                                          collate_fn=utils.collate_fn)
testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          num_workers=4,
                                          shuffle=True,
                                          collate_fn=utils.collate_fn)

num_classes = 2
model = get_model_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=5e-3,
                            momentum=0.9, weight_decay=5e-4)
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=3,
                                               gamma=0.1)

epochs = 10
for epoch in range(epochs):
    train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
    lr_schedular.step()
    evaluate(model, testloader, device)
    
# Checking
img, _ = testset[0]
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
    
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())