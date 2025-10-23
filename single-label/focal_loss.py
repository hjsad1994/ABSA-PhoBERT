"""
Focal Loss for Single-Label Classification
Handles class imbalance by focusing on hard examples

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
https://arxiv.org/abs/1708.02002
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer


class FocalLoss(nn.Module):
    """
    Single-label Focal Loss
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
        p_t = probability of the ground truth class
    
    Args:
        alpha (float or list): Weighting factor(s) for class(es)
                              float: same alpha for all classes
                              list: per-class alpha weights
                              Default: 0.25
        gamma (float): Focusing parameter, controls how much to focus on hard examples
                      Default: 2.0
                      - gamma = 0: equivalent to standard CE
                      - gamma > 0: reduce loss for well-classified examples
        reduction (str): Specifies the reduction to apply to the output
                        Options: 'mean', 'sum', 'none'
    
    Example:
        >>> focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 3)  # batch_size=32, num_classes=3
        >>> targets = torch.randint(0, 3, (32,))  # class indices
        >>> loss = focal_loss(logits, targets)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        
        # Convert alpha to tensor if it's a list
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (Tensor): Logits from model, shape (N, C)
            targets (Tensor): Class indices, shape (N,)
        
        Returns:
            Tensor: Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Compute standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of true class
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if specified
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha: select alpha for each sample's target class
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha[targets]
            else:
                # Single alpha for all classes
                alpha_t = self.alpha
            
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def calculate_focal_alpha(train_labels, task_type="single_label"):
    """
    Calculate optimal focal loss alpha based on class distribution
    
    Alpha balances different classes. Higher alpha for a class = more weight on that class.
    
    Formula:
        For single-label: alpha = median(class_frequencies)
        This balances all classes equally
    
    Args:
        train_labels (ndarray): Training labels, shape (N,) with class indices
        task_type (str): "single_label" (for compatibility with multi-label version)
    
    Returns:
        float: Optimal alpha value in range [0.1, 0.9]
    
    Example:
        >>> # 3 classes: 100 samples class-0, 50 class-1, 20 class-2
        >>> train_labels = np.array([0]*100 + [1]*50 + [2]*20)
        >>> alpha = calculate_focal_alpha(train_labels)
        >>> print(f"Alpha: {alpha:.3f}")  # Alpha: 0.294 (median frequency)
    """
    # Calculate class frequencies
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    
    # Frequency of each class
    class_freq = counts / total
    
    # Use median frequency as alpha (balances all classes)
    alpha = np.median(class_freq)
    
    # Clip to reasonable range [0.1, 0.9]
    alpha = float(np.clip(alpha, 0.1, 0.9))
    
    return alpha


def calculate_per_class_alpha(train_labels, num_classes):
    """
    Calculate per-class alpha weights for focal loss
    
    Per-class alpha gives higher weight to minority classes
    
    Formula:
        alpha[i] = (1 - freq[i]) for class i
        This gives higher alpha to rarer classes
    
    Args:
        train_labels (ndarray): Training labels, shape (N,)
        num_classes (int): Number of classes
    
    Returns:
        ndarray: Per-class alpha weights, shape (num_classes,)
    
    Example:
        >>> train_labels = np.array([0]*100 + [1]*50 + [2]*20)
        >>> alphas = calculate_per_class_alpha(train_labels, 3)
        >>> print(alphas)  # [0.41, 0.71, 0.88] - higher for rarer classes
    """
    # Count samples per class
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    
    # Initialize alpha array
    alphas = np.ones(num_classes)
    
    # Calculate alpha for each class
    for class_idx, count in zip(unique, counts):
        freq = count / total
        # Higher alpha for rarer classes
        alphas[class_idx] = 1.0 - freq
    
    # Normalize to [0.1, 0.9] range
    alphas = alphas / alphas.sum()  # Normalize
    alphas = np.clip(alphas * 3, 0.1, 0.9)  # Scale and clip
    
    return alphas


# ============================================================================
# Custom Trainer with Focal Loss
# ============================================================================

class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer with Focal Loss
    
    Automatically uses Focal Loss instead of default CrossEntropyLoss
    
    Args:
        focal_alpha (float or list): Alpha parameter for Focal Loss
        focal_gamma (float): Gamma parameter for Focal Loss
        *args, **kwargs: Arguments for base Trainer class
    
    Example:
        >>> trainer = FocalLossTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     focal_alpha=0.25,
        ...     focal_gamma=2.0
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, *args, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use Focal Loss
        
        Args:
            model: The model to train
            inputs: Dict with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (optional, for compatibility)
        
        Returns:
            Loss tensor (and outputs if return_outputs=True)
        """
        # Extract labels
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute focal loss
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
