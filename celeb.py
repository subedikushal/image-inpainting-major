import os
import pandas as pd
from skimage import io
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
np.random.seed(31)


def gen_line_mask(size: tuple, thickness_range: tuple = (1, 3), bg_color=0, patch_color: tuple = (255, 255, 255)):
    h, w, _ = size
    mask = np.full(size, bg_color, np.uint8)

    for _ in range(7):
        # get random x locations to start line
        x1, y1 = np.random.randint(1, w-1), np.random.randint(1, h-1)
        # get random y locations to start line
        x2, y2 = np.random.randint(1, w-1), np.random.randint(1, h-1)
        # get random thickness of the line drawn
        thickness = np.random.randint(
            min(1, thickness_range[0]), thickness_range[1])
        # draw black line on hte white mask
        cv2.line(mask, (x1, y1), (x2, y2), patch_color, thickness)
    return mask

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return d


class CelebDataset(Dataset):
    def __init__(self, split, transform):
        self.root_dir = "./dataset"
        self.partition_frame = pd.read_csv("./list_eval_partition.csv")
        self.transform = transform
        self.train = split == 'train'
        self.test = split == 'test'
        self.val = split == 'val'
        self.train_frame = self.partition_frame.loc[self.partition_frame['partition'] == 0]
        self.val_frame = self.partition_frame.loc[self.partition_frame['partition'] == 1]
        self.test_frame = self.partition_frame.loc[self.partition_frame['partition'] == 2]

    def __len__(self):
        if self.train:
            return len(self.train_frame)
        elif self.val:
            return len(self.val_frame)
        elif self.test:
            return len(self.test_frame)

    def __getitem__(self, index):
        mask1 = gen_line_mask((218, 178, 3), (7, 12))
        if self.train:
            img_name = os.path.join(
                self.root_dir, self.train_frame.iloc[index, 0])
            image = io.imread(img_name)
            inp_image = image.copy()
            inp_image[mask1 == 0] = 255
        elif self.val:
            img_name = os.path.join(
                self.root_dir, self.val_frame.iloc[index, 0])
            image = io.imread(img_name)
            inp_image = image.copy()
            inp_image[mask1 == 0] = 255
        elif self.test:
            img_name = os.path.join(
                self.root_dir, self.test_frame.iloc[index, 0])
            image = io.imread(img_name)
            inp_image = image.copy()
            inp_image[mask1 == 0] = 255
        if self.transform:
            return self.transform(inp_image), self.transform(image)
        return inp_image, image


class CelebDatasetFast(Dataset):
    def __init__(self, split, transform, total=200000, mask_size=(178, 178, 3)):
        self.root_dir = "./dataset"
        self.partition_frame = pd.read_csv("./list_eval_partition.csv")
        self.transform = transform
        self.train = split == 'train'
        self.test = split == 'test'
        self.val = split == 'val'
        self.total= total
        self.mask_size = mask_size
        self.train_frame = self.partition_frame.loc[self.partition_frame['partition'] == 0]
        self.val_frame = self.partition_frame.loc[self.partition_frame['partition'] == 1]
        self.test_frame = self.partition_frame.loc[self.partition_frame['partition'] == 2]

    def __len__(self):
        if self.train:
            return 1000
            return len(self.train_frame) 
        elif self.val:
            return 500
            return len(self.val_frame) 
        elif self.test:
            # return 500
            return len(self.test_frame) 

    def __getitem__(self, index):

        mask = torch.from_numpy(gen_line_mask(self.mask_size, (7, 12))).permute((2,0,1))/255

        if self.train:
            img_name = os.path.join(self.root_dir, self.train_frame.iloc[index, 0]) # type: ignore
            image = Image.open(img_name)
        elif self.val:
            img_name = os.path.join(self.root_dir, self.val_frame.iloc[index, 0]) # type: ignore
            image = Image.open(img_name)
        elif self.test:
            img_name = os.path.join(self.root_dir, self.test_frame.iloc[index, 0]) # type: ignore
            image = Image.open(img_name)
        img = self.transform(image)
        return torch.maximum(mask,img),mask,img 
        # return torch.maximum(mask,img),img 

class CelebDatasetNew(Dataset):
    def __init__(self,split, transform):
        self.transform = transform
        self.train = split == 'train'
        self.test = split == 'test'
        self.val = split == 'val'

    def __len__(self):
        if self.train:
            return 20000 
        elif self.test:
            return 5000 
        elif self.val:
            return 5000

    def __getitem__(self, index):
        if self.train:
            inp_name = f"./trainthick/traininput/{index}.png"
            tar_name = f"./trainthick/traintarget/{index}.png"
            inp_image = io.imread(inp_name)
            tar_image = io.imread(tar_name)
        elif self.test:
            inp_name = f"./testthick/testinput/{index}.png"
            tar_name = f"./testthick/testtarget/{index}.png"
            inp_image = io.imread(inp_name)
            tar_image = io.imread(tar_name)
        elif self.val:
            inp_name = f"./validatethick/validateinput/{index}.png"
            tar_name = f"./validatethick/validatetarget/{index}.png"
            inp_image = io.imread(inp_name)
            tar_image = io.imread(tar_name)
        if self.transform:
            return self.transform(inp_image), self.transform(tar_image)
        return inp_image, tar_image

        
if __name__ == "__main__":
    batch_size = 5
    dataset_size = 1000
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Lambda(lambda x: x/255), transforms.Resize([178,178], antialias=True)])
    train_dataset = CelebDatasetFast(
        split='train', transform=transform,total=dataset_size)

    test_dataset = CelebDatasetFast(
        split='test', transform=transform,total=dataset_size)

    val_dataset = CelebDatasetFast(
        split='val', transform=transform,total=dataset_size)

    train_loader = DataLoader(train_dataset, batch_size, True)
    test_loader = DataLoader(test_dataset, batch_size, False)
    val_loader = DataLoader(val_dataset, batch_size, False)
    # print(len(train_loader))
    # print(len(val_loader))
    # train_loader = DataLoader(train_dataset, 10, True)
    # test_loader = DataLoader(test_dataset, 10, False)
    # validate_loader = DataLoader(validate_dataset, 10, False)

    # examples = iter(train_loader)
    # sample = next(examples)
    # inp, target = sample
    # print(inp.shape)

    # train dataset
    # print("Generating Train Dataset")
    # for i in range(10):
    #     inp = train_dataset[i][0].permute((1,2,0)).numpy()*255
    #     target = train_dataset[i][1].permute((1,2,0)).numpy()*255
    #     inp_image = Image.fromarray(inp.astype(np.uint8))
    #     out_image = Image.fromarray(target.astype(np.uint8))
    #     inp_image.save(f'./trainthick/traininput/{i}.png')
    #     out_image.save(f'./trainthick/traintarget/{i}.png')
    #     if i%1000 == 0:
    #         print(i)

    print("Generating Test Dataset")
    for i in range(10):
        inp = test_dataset[i][0].permute((1,2,0)).numpy()*255
        target = test_dataset[i][2].permute((1,2,0)).numpy()*255
        inp_image = Image.fromarray(inp.astype(np.uint8))
        out_image = Image.fromarray(target.astype(np.uint8))
        inp_image.save(f'./testthick/testinput/{i}.png')
        out_image.save(f'./testthick/testtarget/{i}.png')
        if i%1000 == 0:
            print(i)
    # test dataset
    # print("Generating Test Dataset")
    # for i in range(10):
    #     inp = test_dataset[i][0].permute((1,2,0)).numpy()*255
    #     target = test_dataset[i][1].permute((1,2,0)).numpy()*255
    #     inp_image = Image.fromarray(inp.astype(np.uint8))
    #     out_image = Image.fromarray(target.astype(np.uint8))
    #     inp_image.save(f'./testthick/testinput/{i}.png')
    #     out_image.save(f'./testthick/testtarget/{i}.png')
    #     if i%1000 == 0:
    #         print(i)

    # # validate dataset
    # print("Generating Validate Dataset")
    # for i in range(10):
    #     inp = validate_dataset[i][0].permute((1,2,0)).numpy()*255
    #     target = validate_dataset[i][1].permute((1,2,0)).numpy()*255
    #     inp_image = Image.fromarray(inp.astype(np.uint8))
    #     out_image = Image.fromarray(target.astype(np.uint8))
    #     inp_image.save(f'./validatethick/validateinput/{i}.png')
    #     out_image.save(f'./validatethick/validatetarget/{i}.png')
    #     if i%1000 == 0:
    #         print(i)

    # for k in range(0, 6, 2):
    #     i = inp[k].permute((1, 2, 0))
    #     plt.subplot(6, 2, k+1)
    #     plt.imshow(i)
    #     o = target[k].permute((1, 2, 0))
    #     plt.subplot(6, 2, k+2)
    #     plt.imshow(o)
    # plt.show()
