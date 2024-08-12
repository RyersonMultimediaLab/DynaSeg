import mmcv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from Dynamicsegnet.datasets.base import BaseDataset
from Dynamicsegnet.datasets.builder import DATASETS, DATASOURCES, PIPELINES
from Dynamicsegnet.utils import get_root_logger
from PIL import Image
from torchvision.transforms import Compose

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image

def check_array_condition(array):
    condition = (array > 183) & (array < 255)
    if np.any(condition):
        return False
    return True

def collate_fn(batch):
    imgs = []
    labels = []
    for sample in batch:
        img = sample['img']
        label = sample['label']
        img, label = resize_data(img, label)
        imgs.append(img)
        labels.append(label)

    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    return {'img': imgs, 'label': labels}


def resize_data(image, label):
    res = 128  # or any other desired size
    image = TF.resize(image, (res, res), InterpolationMode.BILINEAR)
    label = TF.resize(label, (res, res), InterpolationMode.NEAREST)
    return image, label
########
@DATASETS.register_module()
class CustomCocoEvalDataset(BaseDataset):
    """Custom dataset for contrastive learning methods that forward one view of the
    img at a time (MoCo, SimCLR)."""

    def __init__(self,
                 data_source,
                 img_out_pipeline,
                 ann_fine2coarse='data/fine_to_coarse_dict.pickle',
                 mode='test',
                 thing=True,
                 stuff=True,
                 res=128,
                 # collate_fn=None
                 ):
        assert data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        self.res = res
        self.stuff = stuff
        self.thing = thing
        # self.collate_fn = collate_fn
        self.ann_fine2coarse = ann_fine2coarse
        self.fine2coarse = self.get_fine2coarse(ann_fine2coarse)
        self.img_out_pipeline = Compose(
            [PIPELINES.build(p) for p in img_out_pipeline])
        logger = get_root_logger()
        logger.info(f'{self.__class__.__name__} initialized:\n'
                    f'Output Pipeline: {self.img_out_pipeline}\n')

    def __getitem__(self, idx):
        img, label = self.data_source.get_sample(idx)

        assert isinstance(img, Image.Image) \
            and isinstance(label, Image.Image), \
            ('The output from the data source must be an Img, got: '
             f'{type(img)}. Please ensure that the list file does '
             'not contain labels.')
        img, label = self.transform_data(img, label)

        return dict(idx=idx, img=img, label=label)

    def get_fine2coarse(self, ann_fine2coarse):
        """Map fine label indexing to coarse label indexing."""
        d = mmcv.load(ann_fine2coarse)
        fine_to_coarse_dict = d['fine_index_to_coarse_index']
        fine_to_coarse_dict[255] = -1
        return fine_to_coarse_dict

    def transform_data(self, image, label):
        # TODO: encapsulate it with data pipeline
        # Resize the image and label to a consistent size
        res1 = 640  # or any other desired size
        res2 = 480  # or any other desired size
        # res1 = 480  # or any other desired size
        # res2 = 320  # or any other desired size
        image = TF.resize(image, (res1, res2), InterpolationMode.BILINEAR)
        label = TF.resize(label, (res1, res2), InterpolationMode.NEAREST)
        # 1. Transformation
        image = self.img_out_pipeline(image)
        if check_array_condition(np.array(label)):
            label = self._label_transform(label)

        return image, label

    def _label_transform(self, label):
        """In COCO-Stuff, there are 91 Things and 91 Stuff. 91 Things (0-90)

        => 12 superclasses (0-11) 91 Stuff (91-181) => 15 superclasses (12-26)

        For [Stuff-15], which is the benchmark IIC uses, we only use 15 stuff
        superclasses.
        """
        label = np.array(label)
        # print("the unique elements 1 of the label array are", np.unique(label))
        fine2coarse = np.vectorize(lambda x: self.fine2coarse[x])
        label = fine2coarse(label)  # Map to superclass indexing.
        # print("the unique elements 2 of the label array are", np.unique(label))
        mask = label >= 255  # Exclude unlabelled.

        # Start from zero.
        if self.stuff and not self.thing:
            # This makes all Things categories negative (ignored.)
            label[mask] -= 12
        elif self.thing and not self.stuff:
            # This makes all Stuff categories negative (ignored.)
            mask = label > 11
            label[mask] = -1
        label = torch.LongTensor(label)
        return label

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError

if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CustomCocoEvalDataset(
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            memcached=False,
            mclient_path=None,
            return_label=True),
        img_out_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg)
        ],
        res=128)

    data = dataset[0]
    img = data['img']
    assert img.shape == (3, 128, 128)
    #
    batch_size = 4  # Specify your desired batch size
    num_workers = 2  # Specify the number of workers for data loading


