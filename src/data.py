import numpy as np
from glob import glob
from PIL import Image
from config import cfg
from transforms import all_transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, domain=None, aug_transform=None, id_transform=None):
        super(ImageDataset, self).__init__()
        self.empty = domain is None
        if domain is None:
            self.image_info = np.empty((1, 2), dtype=object)
        else:
            classes = [] if self.empty else sorted([
                path.replace('\\', '/').split('/')[-1]
                for path in glob(f'{cfg.PATHS.DATA_DIR}/{domain}/*')
                if not path.endswith('.txt')
            ])
            assert self.empty or len(classes) == cfg.DATASET.NUM_CLASSES
            self.image_info = np.array([
                [path.replace('\\', '/'), label]
                for label, cls in enumerate(classes)
                for path in sorted(glob(f'{cfg.PATHS.DATA_DIR}/{domain}/{cls}/*'))
            ], dtype=object)
        self.aug_transform = aug_transform
        self.id_transform = id_transform

    def __len__(self):
        return len(self.image_info)
    
    def __getitem__(self, index):
        path, label = self.image_info[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        item = {'index': index, 'path': path, 'label': label}
        if self.aug_transform: item['image_aug'] = self.aug_transform(image)
        if self.id_transform: item['image_id'] = self.id_transform(image)
        return item

    def append(self, info):
        self.image_info = info if self.empty else np.concatenate((self.image_info, info), axis=0)
        self.empty = False
        return self.image_info

    def delete(self, indices):
        self.image_info = np.delete(self.image_info, indices, axis=0)
        return self.image_info

def build_datasets_dataloaders(source, target):
    kwargs = {
        'batch_size':  cfg.DATALOADER.BATCH_SIZE, 
        'num_workers': cfg.DATALOADER.NUM_WORKERS, 
        'pin_memory':  cfg.DATALOADER.PIN_MEMORY,
    }
    datasets = {
        'src_train':         ImageDataset(domain=source, aug_transform=all_transforms['augmentation_labeled']),
        'src_id':            ImageDataset(domain=source,                                                         id_transform=all_transforms['identity']),
        'tgt_unlabeled':     ImageDataset(domain=target, aug_transform=all_transforms['augmentation_unlabeled'], id_transform=all_transforms['identity']),
        'tgt_labeled':       ImageDataset(               aug_transform=all_transforms['augmentation_labeled']),
        'tgt_pseudolabeled': ImageDataset(               aug_transform=all_transforms['augmentation_labeled'],   id_transform=all_transforms['identity']),
        'tgt_test':          ImageDataset(domain=target,                                                         id_transform=all_transforms['identity']),
    }
    dataloaders = {
        'src_train':          DataLoader(datasets['src_train'],         shuffle=True,  drop_last=True,  **kwargs),
        'src_id':             DataLoader(datasets['src_id'],            shuffle=False, drop_last=False, **kwargs),
        'tgt_unlabeled':      DataLoader(datasets['tgt_unlabeled'],     shuffle=True,  drop_last=True,  **kwargs),
        'tgt_unlabeled_full': DataLoader(datasets['tgt_unlabeled'],     shuffle=False, drop_last=False, **kwargs),
        'tgt_labeled':        DataLoader(datasets['tgt_labeled'],       shuffle=True,  drop_last=False, **kwargs),
        'tgt_pseudolabeled':  DataLoader(datasets['tgt_pseudolabeled'], shuffle=True,  drop_last=False, **kwargs),
        'tgt_test':           DataLoader(datasets['tgt_test'],          shuffle=False, drop_last=False, **kwargs),
    }
    return datasets, dataloaders
