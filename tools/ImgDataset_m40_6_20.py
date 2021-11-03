import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms
#import scipy.io as sio

class RandomMultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display'
        #     , 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
        # self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
        #                    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
        #                    '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
        #                    '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
        #                    '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        self.view_coord = []
        self.rand_view_num = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            # stride = int(20/self.num_views) # 12 6 4 3 2 1
            # all_files = all_files[::stride]
            view_coord = np.load(parent_dir+'/'+self.classnames[i]+'/'+set_+'/view.npy')
            self.view_coord.extend(view_coord)
            rand_view_num = np.load(parent_dir+'/'+self.classnames[i]+'/'+set_+'/random_view_num.npy')
            self.rand_view_num.extend(rand_view_num)
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            rand_view_num_new  =[]
            view_coord_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
                rand_view_num_new.append(self.rand_view_num[rand_idx[i]])
                view_coord_new.append(self.view_coord[rand_idx[i]])
            self.filepaths = filepaths_new
            self.rand_view_num = np.array(rand_view_num_new)
            self.view_coord = np.array(view_coord_new)

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        # RD = self.random_view
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        rand_view_num = self.rand_view_num[idx]
        view_coord = self.view_coord[idx]
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), rand_view_num,view_coord, self.filepaths[idx*self.num_views:(idx+1)*self.num_views])


class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display'
        #     , 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
        # self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
        #                    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
        #                    '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
        #                    '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
        #                    '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)

class RandomSingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        # self.classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display'
        #     , 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
        # self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
        #                    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
        #                    '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
        #                    '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
        #                    '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        self.view_coord = []
        self.rand_view_num =[]
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            view_coord = np.load(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/view.npy')
            self.view_coord.extend(view_coord)
            rand_view_num = np.load(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/random_view_num.npy')
            self.rand_view_num.extend(rand_view_num)
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        filepath_new = []
        for i in range(len(self.rand_view_num)):
            filepath_new.extend(self.filepaths[i*20:i*20+self.rand_view_num[i]])
        self.filepaths = filepath_new
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)