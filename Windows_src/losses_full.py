import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# LOVASZ GRAD
# =====================================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard



# LOVASZ SOFTMAX

def lovasz_softmax_flat(probs, labels):
    C = probs.size(1)
    losses = []

    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue

        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if len(losses) == 0:
        return torch.tensor(0.0, device=probs.device)

    return sum(losses) / len(losses)


def lovasz_softmax(probs, labels):
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))
    labels_flat = labels.view(-1)
    return lovasz_softmax_flat(probs_flat, labels_flat)

# DICE LOSS

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality  = torch.sum(probs + targets_onehot, dims)

        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

# FOCAL LOSS

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

# HYBRID LOSS (Focal + Dice + Lovasz)

class HybridLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.focal = FocalLoss(weight=weight)
        self.dice  = DiceLoss()

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss  = self.dice(logits, targets)
        probs = torch.softmax(logits, dim=1)
        lovasz_loss = lovasz_softmax(probs, targets)

        total = (
            0.6 * focal_loss +
            0.2 * dice_loss +
            0.2 * lovasz_loss
        )
        return total
