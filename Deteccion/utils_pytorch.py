from torch.utils.data import Dataset
import os 
import cv2
class BudMaskDataset(Dataset):
    """Grapevine buds segmentation dataset."""
    def __init__(self, img_list, labels, root_dir, transform=False):
        """
        Args:
            img_list (list): List of image names
            labels (dict): dictionary with key=image, value=mask.
            root_dir (string): string with the base path to the data
            transform (bool): Optional preprocessing to be applied
                on a sample.
        """
        self.img_list = img_list
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        #read image
        img_name = os.path.join(self.root_dir, 'images',
                                self.img_list[idx])
        image = cv2.imread(img_name)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        #read mask
        mask_name = os.path.join(self.root_dir, 'masks',
                                self.labels[self.img_list[idx]])
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
        mask = mask.astype(bool).astype(int)
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
