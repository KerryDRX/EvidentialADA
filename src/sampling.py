import math
import logging
import numpy as np
from config import cfg
from uncertainty import uncertainty_calibration


def sorting(stats, order, ascending):
    if order == 'AU' or order == 'EU':
        return sorted(stats, key=lambda x: np.mean(x[{'AU': 5, 'EU': 6}[order]]), reverse=not ascending)

def uncertainty_sampling(model, datasets, dataloaders, totality, sampling_ratio):
    num_samples = math.ceil(totality * sampling_ratio)
    num_samples_first_round = math.ceil(totality * sampling_ratio * cfg.UNCERTAINTY_SAMPLING.KAPPA)

    stats = uncertainty_calibration(model, dataloaders['tgt_unlabeled_full'])
    stats = sorting(stats, order=cfg.UNCERTAINTY_SAMPLING.ORDER.split()[0], ascending=False)[:num_samples_first_round]
    stats = sorting(stats, order=cfg.UNCERTAINTY_SAMPLING.ORDER.split()[1], ascending=False)[:num_samples]
    stats = np.array(stats, dtype=object)

    datasets['tgt_labeled'].append(stats[:, [1,2]])
    datasets['tgt_unlabeled'].delete(stats[:, 0].astype(int))

    return stats

def certainty_sampling(model, datasets, dataloaders, totality, sampling_ratio):
    num_samples = math.ceil(totality * sampling_ratio)
    stats = uncertainty_calibration(model, dataloaders['tgt_unlabeled_full'])
    stats = sorting(stats, order=cfg.CERTAINTY_SAMPLING.ORDER, ascending=True)
    stats = np.array(stats, dtype=object)

    certain_samples = []
    while len(certain_samples) < num_samples:
        selected_classes = set()
        selected_indices = []
        for i in range(stats.shape[0]):
            pred = stats[i, 3]
            if pred not in selected_classes:
                selected_classes.add(pred)
                selected_indices.append(i)
                certain_samples.append(stats[i])
                if len(certain_samples) == num_samples:
                    break
        stats = np.delete(stats, selected_indices, axis=0)
    certain_samples = np.array(certain_samples)

    logging.info(f'Certain Sample Correctness: {(certain_samples[:, 2] == certain_samples[:, 3]).sum()}/{certain_samples.shape[0]}')
    datasets['tgt_pseudolabeled'].append(certain_samples[:, [1, 3]])
    datasets['tgt_unlabeled'].delete(certain_samples[:, 0].astype(int))
