import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from glob import glob
import random
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ConvertImageDtype(torch.float)
])

class GAN_CustomDataset(Dataset):
    def __init__(self, data_paths,transform=None,output_path=True):
        self.data_path_monet, self.data_path_photo = self.get_all_path(data_paths)
        self.transform = transform
        self.output_path = output_path

    def __len__(self):
        return len(self.data_path_photo)
    

    def __getitem__(self, idx):
        sample_monet_paths = self.data_path_monet[idx]
        sample_photo_paths = self.data_path_photo[idx]
    
        sample_monet = []
        sample_photo = []
        if self.output_path == True:
            sample_monet = sample_monet_paths
            sample_photo = sample_photo_paths
            
        elif self.output_path == False:
           
            img_monet = Image.open(sample_monet_paths).convert('RGB') # (height, width, channels)
            img_monet = self.transform(img_monet)
            sample_monet = img_monet

            img_photo = Image.open(sample_photo_paths).convert('RGB') # (height, width, channels)
            img_photo = self.transform(img_photo)
            sample_photo = img_photo

            # Convert list of images to tensor
            sample_monet = torch.tensor(sample_monet)
            sample_photo = torch.tensor(sample_photo) 

        return sample_monet, sample_photo
    
    def get_all_path(self,data_paths):
        path_monet = [i for i in glob(data_paths[0])]
        path_photo = [i for i in glob(data_paths[1])]
        
        # Oversampling
        size_difference = np.abs(len(path_monet) - len(path_photo))
        path_monet += random.choices(path_monet, k=size_difference)
        
        return path_monet, path_photo