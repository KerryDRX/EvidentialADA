import os
import sys
import torch
import logging
import numpy as np
from config import cfg
from model import Model
from loss import EDL_Loss
from tqdm import tqdm, trange
from data import build_datasets_dataloaders
from utils import set_random_seed, LrScheduler, duplicate
from uncertainty import uncertainty, uncertainty_calibration
from sampling import uncertainty_sampling, certainty_sampling


def test(model, dataloader):
    model.eval()
    acc, total = 0, 0
    with torch.no_grad():
        for item in tqdm(dataloader, desc='Test'):
            image, label = item['image_id'].cuda(), item['label']
            with torch.cuda.amp.autocast():
                alpha = model(image)
            pred = alpha.detach().cpu().argmax(dim=1)
            acc += (pred == label).sum().item()
            total += label.shape[0]
    return 100 * acc / total

def train(source, target):
    datasets, dataloaders = build_datasets_dataloaders(source, target)
    iter_per_epoch = max(len(dataloaders['src_train']), len(dataloaders['tgt_unlabeled']))
    max_iters = cfg.TRAINER.MAX_EPOCHS * iter_per_epoch
    totality = len(datasets['tgt_unlabeled'])
    active_round = 0
    best_accuracy = 0

    model = Model().cuda()
    optimizer = torch.optim.SGD(model.parameters_list(), lr=cfg.TRAINER.LR, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = LrScheduler(optimizer, max_iters, init_lr=cfg.TRAINER.LR, gamma=1e-3, decay_rate=0.75)
    criterion = EDL_Loss(loss_fn=cfg.LOSS.FUNCTION, regularization=True)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(1, cfg.TRAINER.MAX_EPOCHS+1):
        # model training
        model.train()

        losses = []
        for batch_idx in trange(iter_per_epoch, desc=f'Train Epoch {epoch}'):
            # create iterators
            if batch_idx % len(dataloaders['src_train']) == 0:
                src_iter = iter(dataloaders['src_train'])
            if batch_idx % len(dataloaders['tgt_unlabeled']) == 0:
                tgt_unlabeled_iter = iter(dataloaders['tgt_unlabeled'])
            if not datasets['tgt_labeled'].empty and batch_idx % len(dataloaders['tgt_labeled']) == 0:
                tgt_labeled_iter = iter(dataloaders['tgt_labeled'])
            if not datasets['tgt_pseudolabeled'].empty and batch_idx % len(dataloaders['tgt_pseudolabeled']) == 0:
                tgt_pseudolabeled_iter = iter(dataloaders['tgt_pseudolabeled'])
            
            # retrieve images and labels
            src_item = next(src_iter)
            tgt_unlabeled_item = next(tgt_unlabeled_iter)
            if not datasets['tgt_labeled'].empty:
                tgt_labeled_item = next(tgt_labeled_iter)
            if not datasets['tgt_pseudolabeled'].empty:
                tgt_pseudolabeled_item = next(tgt_pseudolabeled_iter)

            # model optimization
            scheduler.step()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # source
                loss1 = criterion(model(src_item['image_aug'].cuda()), src_item['label'].cuda())
                # unlabeled target
                AU, EU = uncertainty(model(tgt_unlabeled_item['image_aug'].cuda()), reduce=True)
                loss2 = cfg.LOSS.LAMBDA_AU * AU + cfg.LOSS.LAMBDA_EU * EU
                # labeled target
                loss3 = criterion(model(duplicate(tgt_labeled_item['image_aug']).cuda()), duplicate(tgt_labeled_item['label']).cuda()) if not datasets['tgt_labeled'].empty else 0
                # pseudolabeled target
                loss4 = criterion(model(duplicate(tgt_pseudolabeled_item['image_aug']).cuda()), duplicate(tgt_pseudolabeled_item['label']).cuda()) if not datasets['tgt_pseudolabeled'].empty else 0
                # total loss
                loss = (0.5 * loss1 if cfg.DATASET.NAME == 'Visda-2017' and epoch > 1 else loss1) + loss2 + loss3 + loss4
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        logging.info(f'Epoch {epoch} Train Loss: {np.mean(losses):.3f}')

        # active sampling
        if epoch in cfg.UNCERTAINTY_SAMPLING.EPOCHS:
            certainty_sampling(model, datasets, dataloaders, totality, sampling_ratio=cfg.CERTAINTY_SAMPLING.RATIO[active_round])
            uncertainty_sampling(model, datasets, dataloaders, totality, sampling_ratio=cfg.UNCERTAINTY_SAMPLING.RATIO[active_round])
            active_round += 1

        # model testing
        if epoch % cfg.TRAINER.EVAL_INTERVAL == 0:
            accuracy = test(model, dataloaders['tgt_test'])
            best_accuracy = max(best_accuracy, accuracy)
            
            if cfg.SAVE_MODEL:
                model_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler,
                    'scaler': scaler.state_dict(),
                }
                torch.save(model_checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/checkpoints/{source}_{target}_epoch{epoch:02}_model.pt')
            if cfg.SAVE_STAT:
                tgt_stats = uncertainty_calibration(model, dataloaders['tgt_unlabeled_full'])
                torch.save(tgt_stats, f'{cfg.PATHS.OUTPUT_DIR}/checkpoints/{source}_{target}_epoch{epoch:02}_stat.pt')
            
            logging.info(f'Epoch {epoch} Test Accuracy: {accuracy:.1f} [Best Accuracy: {best_accuracy:.1f}]')
        
    logging.info(f'{source}->{target} Final Accuracy: {accuracy:.1f}')
    logging.info(f'{source}->{target} Best Accuracy: {best_accuracy:.1f}')
    return accuracy, best_accuracy

def main():
    os.makedirs(f'{cfg.PATHS.OUTPUT_DIR}/checkpoints', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=f'{cfg.PATHS.OUTPUT_DIR}/outputs.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f'\n{cfg}\n')
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    final_accuracies, best_accuracies = [], []
    for source in cfg.DATASET.SOURCE_DOMAINS:
        for target in cfg.DATASET.TARGET_DOMAINS:
            if source == target: continue
            logging.info(f'{source} -> {target}')
            set_random_seed(cfg.SEED)
            final_accuracy, best_accuracy = train(source, target)
            final_accuracies.append(final_accuracy)
            best_accuracies.append(best_accuracy)
    logging.info(f'Average Result of {len(final_accuracies)} Domain Transitions:')
    logging.info(f'Average Final Accuracy: {np.mean(final_accuracies):.1f}')
    logging.info(f'Average Best Accuracy: {np.mean(best_accuracies):.1f}')

if __name__ == '__main__':
    main()
