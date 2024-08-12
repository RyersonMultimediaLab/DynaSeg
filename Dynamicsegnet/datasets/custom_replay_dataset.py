import os.path as osp
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms.functional import resize

from Dynamicsegnet.datasets.base import BaseDataset
from Dynamicsegnet.datasets.builder import DATASETS, DATASOURCES, PIPELINES
from Dynamicsegnet.datasets.pipelines import IndexCompose
from Dynamicsegnet.utils import get_root_logger

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image

@DATASETS.register_module(force=True)
class CustomReplayDataset(BaseDataset):
    def __init__(self, data_source, inv_pipelines, shared_pipelines, out_pipeline,
                 mode, prefetch=False, return_label=True, res1=320, res2=640,eqv_pipeline=None,
                 # collate_fn=None
                 ):
        assert not data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        default_args = dict(N=self.data_source.get_length())
        self.reshuffle()
        self.shared_pipeline = IndexCompose([PIPELINES.build(p) for p in shared_pipelines])
        self.inv_pipeline = IndexCompose([PIPELINES.build(p, default_args=default_args) for p in inv_pipelines])
        self.out_pipeline = Compose([PIPELINES.build(p) for p in out_pipeline])
        self.return_label = return_label
        self.prefetch = prefetch
        self.res1 = res1
        self.res2 = res2
        self.mode = mode
        self.eqv_pipeline = eqv_pipeline
        # self.collate_fn = collate_fn

        logger = get_root_logger()
        logger.info(f'{self.__class__.__name__} initialized:\n'
                    f'Shared initial Pipeline:\n{self.shared_pipeline}\n\n'
                    f'Invariant Pipeline:\n{self.inv_pipeline}\n\n'
                    f'Output Pipeline: {self.out_pipeline}\n')

    def reshuffle(self):
        self.shuffled_indices = np.arange(self.data_source.get_length())
        np.random.shuffle(self.shuffled_indices)

    def reset_pipeline_randomness(self):
        self.inv_pipeline.reset_randomness()
        logger = get_root_logger()
        logger.info('Randomness reset for pipelines')

    def __getitem__(self, idx):
        idx = self.shuffled_indices[idx]
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            (f'The output from the data source must be an Image, got {type(img)}. '
             f'Please ensure that the list file does not contain labels.')
        img = self.transform_img(idx, img)
        label = self.transform_label(idx)
        data = dict(idx=idx, img=img)
        if label[0] is not None:
            data.update(label=label)
        return data

    def transform_label(self, idx):
        if hasattr(self, 'return_label') and not self.return_label:
            return (None, )

        label_path = osp.join(self.labeldir, f'{idx}.png')
        label = Image.open(label_path)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return (label, )

    def transform_img(self, idx, img):
        img = self.shared_pipeline(idx, img)
        img = self.inv_pipeline(idx, img)
        img = self.out_pipeline(img)
        return img

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CustomReplayDataset(
        data_source=dict(
            type='CocoImageList',
            root='/home/boujub/PycharmProjects/DynamicSegNet/tools/data/coco',
            img_prefix='train2017',
            seg_prefix='stuffthingmaps/train2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='/home/boujub/PycharmProjects/DynamicSegNet/tools/data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
            # memcached=False,
            # mclient_path=None,
            # return_label=False
        ),
        # inv_pipelines=[
        #     dict(type='ReplayRandomColorBrightness', x=0.3, p=0.8),
        #     dict(type='ReplayRandomColorContrast', x=0.3, p=0.8),
        #     dict(type='ReplayRandomColorSaturation', x=0.3, p=0.8),
        #     dict(type='ReplayRandomColorHue', x=0.1, p=0.8),
        #     dict(type='ReplayRandomGrayScale', p=0.2),
        #     dict(type='ReplayRandomGaussianBlur', sigma=[.1, 2.], p=0.5)
        # ],
        shared_pipelines=[dict(type='ResizeCenterCrop', res=640)],
        out_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg)
        ],
        inv_pipelines=[],
        eqv_pipeline=[],
        prefetch=False,
        mode='compute',
        res1=320,
        res2=640
    )

    dataset.reset_pipeline_randomness()
    data = dataset[0]
    img = data['img']
    assert img.shape == (3, 640, 640)
