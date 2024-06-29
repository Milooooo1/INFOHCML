from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights
from crp.visualization import FeatureVisualization
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import pandas as pd
import torchvision
import torch
import os

def get_train_transformations(norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)

def get_val_transformations(norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)

class HAR(Dataset):
    def __init__(self, img_size, num_classes, dataset_dir, train=True, transform=None):
        self.size = img_size
        self.num_classes = num_classes
        self.train = train
        self.transform = transform
        
        # Load the dataset from the CSV file !! UPDATE THIS LINE IF
        data = pd.read_csv(dataset_dir + "/data.csv")
        
        # Select the subset of the data based on the split
        if self.train:
            self.data = data[data['split'] == 0]
        else:
            self.data = data[data['split'] == 1]
        
        self.image_names = self.data['image'].values
        self.labels = self.data['label'].values
        self.images_folder = dataset_dir + "/train/"
        
        # Create a label mapping to ensure consistent label numbers
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.inverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        self.mapped_labels = [self.label_mapping[label] for label in self.labels]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.mapped_labels[idx], dtype=torch.long)
        
        return image, label

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=15):
        super(EfficientNetModel, self).__init__()
        # Load pre-trained EfficientNet model
        self.efficientnet = efficientnet_v2_m(weights = EfficientNet_V2_M_Weights.DEFAULT)

        for name, param in self.efficientnet.named_parameters():
            param.requires_grad = False


        # Replace the classifier (head) of EfficientNet
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

import torch
import numpy as np
# from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

data_path = r"C:\Users\Milo\OneDrive - Universiteit Utrecht\Period 4\INFOMHCML\Project\BotCL\data\HAR\Human Action Recognition" #TODO fill or run download below
checkpoint_path = r"C:\Users\Milo\OneDrive - Universiteit Utrecht\Period 4\INFOMHCML\Project\zennit-crp\tutorials\checkpoints\EfficientNetV2M_Acc74.pth"

weights = EfficientNet_V2_M_Weights.DEFAULT
preprocess = weights.transforms()
HAR_dataset = HAR(img_size=160, num_classes=15, dataset_dir = data_path, transform=preprocess)

model = EfficientNetModel().to("cpu")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlusFlat(canonizers)

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
denormalize = T.Normalize(
    mean=-mean / std, 
    std= 1.0 / std
)

transform = T.ToPILImage()
img = denormalize(HAR_dataset[np.random.randint(0, len(HAR_dataset))][0])
# img = transform(img)
# img


cc = ChannelConcept()

layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
layer_map = {layer : cc for layer in layer_names}

attribution = CondAttribution(model)

# separate normalization from resizing for plotting purposes later
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

fv_path = "EfficientNet"
fv = FeatureVisualization(attribution, HAR_dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

saved_files = fv.run(composite, 0, len(HAR_dataset), 32, 100)