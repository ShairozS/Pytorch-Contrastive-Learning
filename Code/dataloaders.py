from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

class LabeledContrastiveDataset(Dataset):
    """
    Take a folder containing sub-folders of images, where the sub-folder name is the image class, and generate
    pairs of images with the same class label for contrastive learning. This procedure fixes the batch size since
    every batch contains all classes and every batch element is a unique pairing of each class. 
    """

    def __init__(self, folder, transforms=None):
        
        labels = os.listdir(folder)
        
        self.labels_to_imgs = dict() # Dict with labels as keys and file paths as values
        self.idx_to_imgs = [] # List with tuples of each filename and label (label, fname)
        self.transform = transforms
        
        

        for label in labels: # Populate 
            current_dir = os.path.join(folder, label)
            label_files = [os.path.join(current_dir, x) for x in os.listdir(current_dir)]
            self.labels_to_imgs[label] = label_files
            self.idx_to_imgs += [(f, label) for f in label_files]
    
    def __len__(self):
        return(len(self.idx_to_imgs))
    
    def __getitem__(self, idx):
        '''
        
        
        '''
        # Grab the index image as the anchor
        img, label =  self.idx_to_imgs[idx]
        img = plt.imread(img)
        label_tensor = []
        
        # Grab an image with the same class as the anchor if available
        similar_imgs = len(self.labels_to_imgs[label])
        if similar_imgs > 2:
            similar_imgs_idx = np.random.choice(range(similar_imgs))
            similar_img = self.labels_to_imgs[label][similar_imgs_idx]
            similar_img = plt.imread(similar_img)
        else:
            raise NotImplementedError
        
        
        if self.transform is not None:
            img = self.transform(img); similar_img = self.transform(similar_img)
        
        
        out_tensor_x1 = img[np.newaxis,...]
        out_tensor_x2 = similar_img[np.newaxis,...]
        label_tensor.append(int(label))
        
        # To form the batch, grab all the other classes 
        for l in ((set(self.labels_to_imgs.keys())) - set(label)):
            
            dissimilar_imgs = len(self.labels_to_imgs[l])
            dissimilar_img_idx = np.random.choice(range(dissimilar_imgs))
            dissimilar_img = self.labels_to_imgs[l][dissimilar_img_idx]
            dissimilar_img = plt.imread(dissimilar_img)
            #dissimilar_img = np.expand_dims(dissimilar_img, 0)
            
            dissimilar_img_idx2 = np.random.choice(range(dissimilar_imgs))
            while dissimilar_img_idx2 == dissimilar_img_idx:
                dissimilar_img_idx2 = np.random.choice(range(dissimilar_imgs))

            dissimilar_img2 = self.labels_to_imgs[l][dissimilar_img_idx2]
            dissimilar_img2 = plt.imread(dissimilar_img2)
            #dissimilar_img2 = np.expand_dims(dissimilar_img2, 0)
    
            if self.transform is not None:
                dissimilar_img = self.transform(dissimilar_img)
                dissimilar_img2 = self.transform(dissimilar_img2)
            
            dissimilar_img = dissimilar_img[np.newaxis,...]
            dissimilar_img2 = dissimilar_img2[np.newaxis,...]
            
            out_tensor_x1 = torch.cat([out_tensor_x1, dissimilar_img])
            out_tensor_x2 = torch.cat([out_tensor_x2, dissimilar_img2])
            
            label_tensor.append(int(l))
        
        out_dict = {"x1": out_tensor_x1, "x2": out_tensor_x2, "labels": torch.Tensor(label_tensor)}
        return(out_dict)