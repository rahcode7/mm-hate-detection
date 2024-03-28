## Load FHM dataset
from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image
import json 
from icecream import ic 

class FBHMDataset(Dataset):
    def __init__(self,root_dir,split) -> None:
        splitfile = split + '.jsonl'
        data = []
        
        FILE_PATH = os.path.join(root_dir,splitfile)
        with open(FILE_PATH,'r',encoding='utf8') as f:
            for line in f:
                data.append(json.loads(line))
        self.lb = data
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.lb)

    def __getitem__(self,idx):
        file_name = str(self.lb[idx]["img"]) # + ".png"
        
        label = self.lb[idx]["label"]
        #ic(file_name,idx,label,self.lb[idx]["id"])
        
        image_path = os.path.join(self.root_dir,file_name)
        image = Image.open(image_path).convert('RGB')  

        return file_name,image,label

def collate_fn(batch):
    file_list,image_list,label_list = [],[],[]

    batch = list(filter (lambda x:x is not None, batch))
    
    for file,image,label in batch:
        file_list.append(file)
        image_list.append(image)
        label_list.append(label)    
    return file_list,image_list,label_list




