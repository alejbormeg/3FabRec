import os
import numpy as np
import pandas as pd
import torch.utils.data as td
from csl_common.vis import vis
from csl_common.utils import geometry
from facenet_pytorch import MTCNN, InceptionResnetV1
from datasets import facedataset
import re
from skimage import io
import random
import ast
from csl_common.utils import ds_utils

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class FORENSE_AM(facedataset.FaceDataset):
    NUM_LANDMARKS = 30
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))
    LANDMARKS_NO_OUTLINE = ALL_LANDMARKS  # no outlines in AFLW
    LANDMARKS_ONLY_OUTLINE = ALL_LANDMARKS  # no outlines in AFLW

    def __init__(self, root, cache_root=None, test_split=None, landmark_ids=range(30), cross_val_split=None, **kwargs):
        self.cross_val_split = cross_val_split
        super().__init__(root=root,
                         cache_root=cache_root,
                         test_split=test_split,
                         landmark_ids=landmark_ids,
                         fullsize_img_dir=root,
                         **kwargs)

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.face_h.values

    @property
    def widths(self):
        return self.annotations.face_w.values

    def _load_annotations(self,split):

        return self.make_split(self.cross_val_split,split)

    def make_split(self, cross_val_split,split):
        df = pd.read_csv(os.path.join(self.root, 'annotations.csv'),index_col=0)
        ids = list(range(131))
        annotations=df
        print("\t\t\tPARAMETROS")
        print("\t\t\tCross_val_split", cross_val_split)
        print("\t\t\tSplit", split)

        #Particion Final
        if(cross_val_split==None):
            df = pd.read_csv(os.path.join(self.root, 'annotations.csv'), index_col=0)
            train_ids=[32, 94, 36, 105, 119, 86, 6, 7, 75, 108, 50, 23, 51, 79, 69, 54, 59, 127, 39, 158, 110, 26, 102, 65, 156, 95, 115, 128, 140, 106, 13, 161, 142, 29, 48, 88, 99, 129, 93, 101, 42, 150, 33, 78, 43, 87, 81, 134, 37, 25, 103, 16, 5, 154, 62, 138, 2, 45, 113, 0, 117, 27, 104, 80, 131, 73, 89, 82, 141, 58, 151, 15, 22, 66, 18, 143, 31, 130, 53, 111, 136, 124, 21, 56, 12, 14, 91, 76, 47, 139, 49, 148, 10, 160, 9, 84, 135, 121, 159, 100, 4, 122, 157, 28, 20, 64, 153, 74, 149, 152, 60, 55, 57, 70, 85, 40, 126, 114, 98, 46, 68, 38, 132, 120, 24, 17, 30, 1, 145, 96, 112]
            test_ids=[]
            for i in range(164):
                if i not in train_ids:
                    test_ids.append(i)

            if(split=='train'):
                annotations=df[df.ra.isin(train_ids)]
            elif (split== 'test'):
                annotations=df[df.ra.isin(test_ids)]

        #primer 20% de los datos para test y 80% restante para train
        elif(cross_val_split==1):
            if (split == 'train'):
                train_ids = ids[int(len(ids) * 0.2):]
                annotations = df[df.ra.isin(train_ids)]
            elif (split == 'test'):
                test_ids = ids[0:int(len(ids) * 0.2)]
                annotations = df[df.ra.isin(test_ids)]
        # segundo 20% de los datos para test y 80% restante para train
        elif (cross_val_split ==2):
            if (split == 'train'):
                list1 = ids[0:int(len(ids) * 0.2)]
                list2 = ids[int(len(ids) * 0.4):]
                train_ids = list1 + list2
                annotations = df[df.ra.isin(train_ids)]
            elif (split == 'test'):
                test_ids = ids[int(len(ids) * 0.2):int(len(ids) * 0.4)]
                annotations = df[df.ra.isin(test_ids)]

        # tercer 20% de los datos para test y 80% restante para train
        elif (cross_val_split ==3):
            if (split == 'train'):
                list1 = ids[0:int(len(ids) * 0.4)]
                list2 = ids[int(len(ids) * 0.6):]
                train_ids = list1 + list2
                annotations = df[df.ra.isin(train_ids)]
            elif (split == 'test'):
                test_ids = ids[int(len(ids) * 0.4):int(len(ids) * 0.6)]
                annotations = df[df.ra.isin(test_ids)]
        # cuarto 20% de los datos para test y 80% restante para train
        elif (cross_val_split ==4):
            if (split == 'train'):
                list1 = ids[0:int(len(ids) * 0.6)]
                list2 = ids[int(len(ids) * 0.8):]
                train_ids = list1 + list2
                annotations = df[df.ra.isin(train_ids)]
            elif (split == 'test'):
                test_ids = ids[int(len(ids) * 0.6):int(len(ids) * 0.8)]
                annotations = df[df.ra.isin(test_ids)]
        # quinto 20% de los datos para test y 80% restante para train
        elif (cross_val_split ==5):
            if (split == 'train'):
                train_ids = ids[0:int(len(ids) * 0.8)]
                annotations = df[df.ra.isin(train_ids)]
            elif (split == 'test'):
                test_ids = ids[int(len(ids) * 0.8):]
                annotations = df[df.ra.isin(test_ids)]
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        face_id = sample.ra
        bb = [sample.face_x, sample.face_y, sample.face_x + sample.face_w, sample.face_y + sample.face_h]
        landmarks = sample.landmarks_full
        landmarks=ast.literal_eval(landmarks)
        landmarks=np.array(landmarks)
        masks=np.array(ast.literal_eval(sample.masks))
        landmarks_for_crop = landmarks if self.crop_source == 'lm_ground_truth' else None
        real_sample=self.get_sample(sample.fnames, bb, landmarks_for_crop=landmarks_for_crop, id=face_id,
                                    landmarks_to_return=landmarks, mask=masks)
        return  real_sample


import config
config.register_dataset(FORENSE_AM)


if __name__ == '__main__':

    from csl_common.utils.nn import Batch, denormalize
    import csl_common.utils.common
    import torch

    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    csl_common.utils.common.init_random()

    ds = FORENSE_AM(root='/home/alejandro/Escritorio/5ºDGIIM/Segundo Cuatrimestre/TFG/Material_Informatica/datasets/FORENSE_AM_TRAIN',Train=False,test_split='test', deterministic=True, use_cache=True, image_size=256,crop_source='bb_ground_truth',transform=None,cross_val_split=1)
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for data in dl:
        batch = Batch(data, gpu=False)
        print(batch.images.shape)
        inputs = batch.images.clone()
        denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy(), radius=3, color=(0,255,0))
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False,wait=10000)
