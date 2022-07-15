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

    def __init__(self, root, cache_root=None, test_split='full', landmark_ids=range(30), **kwargs):

        assert test_split in ['full', 'frontal']
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
        return self.make_split(self.train)

    def make_split(self, train):
        df = pd.read_csv(os.path.join(self.root, 'annotations.csv'),index_col=0)
        #Vamos a hacer 80% entrenamiento 20% test
        ids=list(range(164))
        random.shuffle(ids)
        train_ids=ids[0:int(len(ids)*0.8)]
        annotations=df[df.ra.isin(train_ids)]
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
        landmarks_for_crop = landmarks if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(sample.fnames, bb, landmarks_for_crop=landmarks_for_crop, id=face_id,
                                    landmarks_to_return=landmarks)



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

    ds = FORENSE_AM(root='/home/alejandro/Escritorio/5ºDGIIM/Segundo Cuatrimestre/TFG/Material_Informatica/datasets/FORENSE_AM',train=True, deterministic=True, use_cache=True, image_size=256,crop_source='bb_ground_truth')
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy(), radius=3, color=(0,255,0))
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False)
