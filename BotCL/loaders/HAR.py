from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import os


class HAR(Dataset):
    def __init__(self, args, train=True, transform=None):
        self.size = args.img_size
        self.num_classes = args.num_classes
        self.train = train
        self.transform = transform
        
        # Load the dataset from the CSV file !! UPDATE THIS LINE IF
        data = pd.read_csv(args.dataset_dir + "data.csv")
        
        # Select the subset of the data based on the split
        if self.train:
            self.data = data[data['split'] == 0]
        else:
            self.data = data[data['split'] == 1]
        
        self.image_names = self.data['image'].values
        self.labels = self.data['label'].values
        self.images_folder = args.dataset_dir + "train/"
        
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