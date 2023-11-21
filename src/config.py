from yacs.config import CfgNode as CN


_C = CN()
_C.SEED = 0
_C.SAVE_MODEL = False
_C.SAVE_STAT = False
_C.UNCERTAINTY = 'variance'  # variance/entropy

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 4

_C.LOSS = CN()
_C.LOSS.FUNCTION = 'nll'  # nll/ce/sos
_C.LOSS.ACTIVATION = 'exp'  # relu/exp/softplus
_C.LOSS.LAMBDA_AU = 0.05

_C.UNCERTAINTY_SAMPLING = CN()
_C.UNCERTAINTY_SAMPLING.EPOCHS = [10, 12, 14, 16, 18]
_C.UNCERTAINTY_SAMPLING.ORDER = 'EU AU'
_C.UNCERTAINTY_SAMPLING.RATIO = [0.01, 0.01, 0.01, 0.01, 0.01]
_C.UNCERTAINTY_SAMPLING.KAPPA = 10

_C.CERTAINTY_SAMPLING = CN()
_C.CERTAINTY_SAMPLING.ORDER = 'EU'
_C.CERTAINTY_SAMPLING.RATIO = [0.01, 0.02, 0.03, 0.04, 0.05]

_C.DATASET = CN()
_C.DATASET.NAME = 'Office-Home'  # Office-Home/Visda-2017

_C.PATHS = CN()
_C.TRAINER = CN()

if _C.DATASET.NAME == 'Office-Home':
    _C.PATHS.DATA_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/OfficeHomeDataset_10072016'
    _C.PATHS.OUTPUT_DIR = 'outputs'
    _C.DATASET.NUM_CLASSES = 65
    _C.DATASET.SOURCE_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']
    _C.DATASET.TARGET_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']
    _C.TRAINER.LR = 4e-3
    _C.TRAINER.MAX_EPOCHS = 50
    _C.TRAINER.EVAL_INTERVAL = 5
    _C.LOSS.LAMBDA_EU = 1.0 if _C.UNCERTAINTY == 'entropy' else 50.0
elif _C.DATASET.NAME == 'Visda-2017':
    _C.PATHS.DATA_DIR = ...
    _C.PATHS.OUTPUT_DIR = ...
    _C.DATASET.NUM_CLASSES = 12
    _C.DATASET.SOURCE_DOMAINS = ['train']
    _C.DATASET.TARGET_DOMAINS = ['validation']
    _C.TRAINER.LR = 1e-3
    _C.TRAINER.MAX_EPOCHS = 40
    _C.TRAINER.EVAL_INTERVAL = 2
    _C.LOSS.LAMBDA_EU = 1.0 if _C.UNCERTAINTY == 'entropy' else 10.0
else:
    raise NotImplementedError(f'Dataset not implemented: {_C.DATASET.NAME}')


def get_cfg_defaults():
    return _C.clone()


cfg = _C
