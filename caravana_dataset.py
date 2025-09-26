import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CaravanaDataset(Dataset): # Doc torch.utils.data.Dataset => Dataset est une classe vide qui représente un dataset, on doit définir len et getitem pour pouvoir utiliser Dataloader par la suite 
    def __init__(self,root_path,test=False): # Init contien les data :
        self.root_path = root_path
        print("loading data ...")
        self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")]) # train images
        self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")]) # train mask
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])                                                                                      # transformations des données, resize et ToTensor() pour pouvoir entrer dans le model 
        print("loading ended ...")

    def __getitem__(self, index): # Retourne l'image et le mask associés à l'index
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L") # Niveau de gris just stores the Luminance
        return self.transform(img), self.transform(mask)
    
    def __len__(self): # Retourne la taille du dataset (Combien il y a d'éléments dans le dataset)
        return len(self.images)
    
# Grâce à Dataloader par la suite on pourra parcourir facilement le dataset (batches, shuffle, ...)