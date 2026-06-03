import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import csv

# Cache for domain mapping across dataset instantiations
_img_to_country_id = None
_num_domains = 12

def _load_domain_data():
    global _img_to_country_id, _num_domains
    if _img_to_country_id is not None:
        return _img_to_country_id, _num_domains
        
    _img_to_country_id = {}
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'path_to_gwhd_2021_here')
    metadata_path = os.path.join(base_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.csv not found at {metadata_path}")
        return {}, 0
        
    domain_to_country = {}
    unique_countries = set()
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            domain_to_country[row['name']] = row['country']
            unique_countries.add(row['country'])
            
    countries = sorted(list(unique_countries))
    country_to_id = {c: i for i, c in enumerate(countries)}
    _num_domains = len(countries)
    
    for split in ['train.csv', 'val.csv', 'test.csv']:
        csv_path = os.path.join(base_dir, split)
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    domain = row['domain']
                    if domain in domain_to_country:
                        country = domain_to_country[domain]
                        _img_to_country_id[row['image_name']] = country_to_id[country]
                        
    return _img_to_country_id, _num_domains
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_to_country_id, self.num_domains = _load_domain_data()
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        if self.transform is not None:
            img = self.transform(img)
            
        filename = os.path.basename(img_path)
        country_id = self.img_to_country_id.get(filename, 0)
        
        return img, target, country_id
