import os
import tarfile
import subprocess
import sys
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def parse_entry(filepath, filename):
    # Load image and check position and classname of RoI
    # At this time, convert classname to label(integer type).
    # The reason it starts from 1 is that 0 is set as the label of the background.
    data = pd.read_csv(filepath)
    boxes_array = data[data["frame"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    classnames = data[data["frame"] == filename][["class_id"]]
    classes = []
    for i in range(len(classnames)):
        classes.append(int(classnames.iloc[i, 0]))
    return boxes_array, classes


class DrinksDataset(torch.utils.data.Dataset):
    # Class for creating dataset and importing dataset into the Datalader
    # Transforms means whether or not the image is preprocessed (left/right transform, etc.)
    def __init__(self, root, df_path, transforms=None):
        self.root = root
        self.transforms = transforms
        self.df = df_path
        names = pd.read_csv(df_path)[['frame']]
        names = names.drop_duplicates()
        self.imgs = list(np.array(names['frame'].tolist()))

    def __getitem__(self, idx):
        # Load image and check image information
        img_path = os.path.join(self.root, self.imgs[idx])
        # Open the specified image
        img = Image.open(img_path).convert("RGB")
        # get data
        bbox_list, classes = parse_entry(self.df, self.imgs[idx])
        # Convert to format suitable for learning(torch.tensor type)
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        # get area
        area_list = []
        for i in bbox_list:
            area_list.append((i[2] - i[0]) * (i[3] - i[1]))
        areas = torch.as_tensor(area_list, dtype=torch.float32)
        # assumption is that no crowds
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        # create actual dictionary

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
   transforms = []
   # Converts the image, to a PyTorch Tensor
   transforms.append(T.ToTensor())
   if train:
      # Transform the image left and right with 50% probability when learning
      transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # Load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # Replace the classifier with a new one, that has num_classes which is user-defined
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


path = os.getcwd()
dataset = 'Drinks'

if not os.path.isdir(dataset):
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown'])
    import gdown
    output = 'drinks.tar.gz'
    url = 'https://drive.google.com/u/0/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA&'
    gdown.download(url, output, quiet=False)
    tar = tarfile.open(output)
    tar.extractall(dataset)
    tar.close()

path_to_data = os.path.join(path, dataset, 'drinks')

dataset_train = DrinksDataset(path_to_data, os.path.join(path_to_data, 'labels_train.csv'), transforms = get_transform(train=True))
dataset_val = DrinksDataset(path_to_data, os.path.join(path_to_data, 'labels_test.csv'), transforms = get_transform(train=False))

# Randomly reorder images in a dataset
torch.manual_seed(1)
indices_train = torch.randperm(len(dataset_train)).tolist()
indices_val = torch.randperm(len(dataset_val)).tolist()
dataset_train = torch.utils.data.Subset(dataset_train, indices_train)
dataset_val = torch.utils.data.Subset(dataset_val, indices_val)

# Define Dataloader
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=1,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda:0')
num_classes = 4  # 3 class (number of classname) + 1 class (background)
model = get_instance_segmentation_model(num_classes)
# Move model to GPU or CPU
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    # Train for 1 epoch and print every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # Update learning rate
    lr_scheduler.step()
    # Evaluate on the validation data
    evaluate(model, data_loader_val, device=device)

torch.save(model.state_dict(), 'model_weights.pth')
