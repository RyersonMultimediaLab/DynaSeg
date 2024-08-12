import os.path as osp
import sys

import numpy as np
import torch
from Dynamicsegnet.datasets.base import BaseDataset
from Dynamicsegnet.datasets.builder import DATASETS, DATASOURCES, PIPELINES
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image


@DATASETS.register_module(force=True)
class ClusterReplayDatasetDynCnn(BaseDataset):
    """Dataset for contrastive learning methods that forward two views of the
    img at a time (MoCo, SimCLR)."""

    def __init__(self,
                 data_source,
                 mode,
                 prefetch=False,
                 return_label=True):
        assert not data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        default_args = dict(N=self.data_source.get_length())
        self.reshuffle()


        self.return_label = return_label



    def reshuffle(self):
        """Generate random floats for all img data to deterministically random
        transform.

        This is to use random sampling but have the same samples during
        clustering and training within the same epoch.
        """
        self.shuffled_indices = np.arange(self.data_source.get_length())
        np.random.shuffle(self.shuffled_indices)



    def __getitem__(self, idx):
        idx = self.shuffled_indices[idx]
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            (f'The output from the data source must be an Img, got {type(img)}'
             '. Please ensure that the list file does not contain labels.')
        img = self.transform_img(idx, img)
        label = self.transform_label(idx)
        data = dict(idx=idx, img=img)
        if label[0] is not None:
            data.update(label=label)
        return data

    def transform_label(self, idx):
        if hasattr(self, 'return_label') and not self.return_label:
            return (None, )

        # TODO Equiv. transform.
        if self.mode == 'train':
            # assume labels are saved as torch tensor
            # This should be consistent with the PiCIEHook
            label1_path = osp.join(self.labeldir, 'label_1', f'{idx}.png')
            label2_path = osp.join(self.labeldir, 'label_2', f'{idx}.png')
            # should avoid memcache here because the value of labels
            # always change after each epoch
            label1 = Image.open(label1_path)
            label2 = Image.open(label2_path)
            label1 = np.array(label1)
            label2 = np.array(label2)

            label1 = torch.from_numpy(label1).long()
            label2 = torch.from_numpy(label2).long()

            return label1, label2

        elif self.mode == 'baseline_train':
            label1_path = osp.join(self.labeldir, 'label_1', f'{idx}.png')
            label1 = Image.open(label1_path)
            label1 = np.array(label1)
            label1 = torch.from_numpy(label1).long()

            return (label1, )

        return (None, )

    def transform_img(self, img):
        return(img)


    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ClusterReplayDatasetDynCnn(
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='train2017',
            seg_prefix='stuffthingmaps/train2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
            memcached=False,
            mclient_path=None,
            return_label=False),



        prefetch=False,
        mode='compute')

    dataset.view = 1
    data = dataset[0]
    img = data['img'][0]
    assert img.shape == (3, 640, 640)
    dataset.reset_pipeline_randomness()
