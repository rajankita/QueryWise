import numpy as np
import torch
import torch as torch
import torch
import torch.utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import sys
from tqdm import tqdm



def compute_adjustment(train_loader, tro, num_classes):
    """compute the base probabilities"""

    label_freq = {}
    
    # set all freqs to 0
    for i in range(num_classes):
        label_freq[i] = 0
    
    # compute label frequencies from train data
    for i, data in enumerate(train_loader):
        target = data[1].cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments



def get_uncertainty_entropy(model, thief_data, unlabeled_idxs, batch_size=128):
    
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    indexes = torch.tensor([])
    
    unlabeled_loader = DataLoader(Subset(thief_data, unlabeled_idxs), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    with torch.no_grad():
        for data in tqdm(unlabeled_loader):
            inputs = data[0].cuda()
            labels = data[1]
            ind = data[2]
            scores = model(inputs)
            prob_dist = F.softmax(scores).detach().cpu().numpy()
            prbslogs = prob_dist * np.log2(prob_dist + sys.float_info.epsilon)
            numerator = 0 - np.sum(prbslogs, 1)
            denominator = np.log2(prob_dist.shape[1])
            entropy = numerator / denominator
            uncertainty = torch.cat((uncertainty, torch.tensor(entropy)), 0)
            indexes = torch.cat((indexes, ind), 0)

    return uncertainty, indexes