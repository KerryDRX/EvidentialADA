from PIL import Image
from config import cfg
from torchvision import transforms as T


resize = T.Resize((256, 256), interpolation=Image.BICUBIC)

random_resized_crop = T.RandomResizedCrop((224, 224), interpolation=Image.BILINEAR)
random_crop = T.RandomCrop((224, 224))
center_crop = T.CenterCrop((224, 224))
random_horizontal_flip = T.RandomHorizontalFlip(p=0.5)
color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if cfg.DATASET.NAME == 'Office-Home':
    all_transforms = {
        'augmentation_labeled': T.Compose([
            resize,
            random_resized_crop, random_horizontal_flip, color_jitter,
            to_tensor, normalize,
        ]),
        'augmentation_unlabeled': T.Compose([
            resize,
            # random_crop,
            random_resized_crop, random_horizontal_flip,
            to_tensor, normalize,
        ]),
        'identity': T.Compose([
            resize,
            center_crop,
            to_tensor, normalize,
        ]),
    }
if cfg.DATASET.NAME == 'Visda-2017':
    all_transforms = {
        'augmentation_labeled': T.Compose([
            resize,
            center_crop, random_horizontal_flip, color_jitter,
            to_tensor, normalize,
        ]),
        'augmentation_unlabeled': T.Compose([
            resize,
            center_crop, random_horizontal_flip,
            to_tensor, normalize,
        ]),
        'identity': T.Compose([
            resize,
            center_crop,
            to_tensor, normalize,
        ]),
    }
