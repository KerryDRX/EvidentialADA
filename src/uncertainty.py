import torch
from tqdm import tqdm
from config import cfg


def uncertainty(alpha, reduce=False):
    if cfg.UNCERTAINTY == 'variance':
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        variance = p - p ** 2
        EU = (alpha / S) * (1 - alpha / S) / (S + 1)
        AU = variance - EU
        if reduce:
            AU = AU.sum() / alpha.shape[0]
            EU = EU.sum() / alpha.shape[0]
        return AU, EU
    elif cfg.UNCERTAINTY == 'entropy':
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        entropy = - (p * (p + 1e-7).log()).sum(dim=1)
        Udata = ((alpha / S) * ((S + 1).digamma() - (alpha + 1).digamma())).sum(dim=1)
        Udist = entropy - Udata
        if reduce:
            Udata = Udata.sum() / alpha.shape[0]
            Udist = Udist.sum() / alpha.shape[0]
        return Udata, Udist
    else:
        raise NotImplementedError(f'Uncertainty not implemented: {cfg.UNCERTAINTY}')
    

def uncertainty_calibration(model, dataloader):
    model.eval()
    stats = list()
    with torch.no_grad():
        for item in tqdm(dataloader, desc='Uncertainty Calibration'):
            image = item['image_id'].cuda()
            alpha = model(image)
            AU, EU = uncertainty(alpha, reduce=False)
            pred = alpha.argmax(dim=1)
            for i in range(AU.shape[0]):
                stats.append([
                    item['index'][i].item(),
                    item['path'][i],
                    item['label'][i].item(), 
                    pred[i].detach().cpu().item(),
                    alpha[i].detach().cpu().tolist(),
                    AU[i].detach().cpu().tolist(),
                    EU[i].detach().cpu().tolist(),
                ])
    return stats
