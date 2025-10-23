"""
Focal Loss for Multi-Label Classification
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
    Multi-label Focal Loss
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
        p_t = p    if y = 1 (positive class)
        p_t = 1-p  if y = 0 (negative class)
    
    Args:
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples
                      Default: 0.25 (more weight on positive class)
        gamma (float): Focusing parameter, controls how much to focus on hard examples
                      Default: 2.0
                      - gamma = 0: equivalent to standard BCE
                      - gamma > 0: reduce loss for well-classified examples
        reduction (str): Specifies the reduction to apply to the output
                        Options: 'mean', 'sum', 'none'
    
    Example:
        >>> focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 33)  # batch_size=32, num_labels=33
        >>> targets = torch.randint(0, 2, (32, 33)).float()
        >>> loss = focal_loss(logits, targets)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (Tensor): Logits from model, shape (N, C)
            targets (Tensor): Binary labels (0 or 1), shape (N, C)
        
        Returns:
            Tensor: Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Focal loss for positive samples: -alpha * (1-p)^gamma * log(p)
        pos_loss = -self.alpha * torch.pow(1 - probs, self.gamma) * torch.log(probs)
        
        # Focal loss for negative samples: -(1-alpha) * p^gamma * log(1-p)
        neg_loss = -(1 - self.alpha) * torch.pow(probs, self.gamma) * torch.log(1 - probs)
        
        # Combine: use pos_loss when target=1, neg_loss when target=0
        loss = targets * pos_loss + (1 - targets) * neg_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    
    Handles class imbalance by weighting positive samples based on their frequency
    
    Args:
        pos_weights (Tensor): Weights for positive class per label, shape (C,)
                             Higher weight = more importance on positive class
                             Typically: num_negatives / num_positives
        reduction (str): Specifies the reduction to apply
    
    Example:
        >>> pos_weights = torch.tensor([2.0, 1.5, 3.0, ...])  # 33 weights
        >>> weighted_bce = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
        >>> loss = weighted_bce(logits, targets)
    """
    
    def __init__(self, pos_weights=None, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute weighted BCE loss
        
        Args:
            inputs (Tensor): Logits from model, shape (N, C)
            targets (Tensor): Binary labels, shape (N, C)
        
        Returns:
            Tensor: Weighted BCE loss
        """
        if self.pos_weights is not None:
            # Ensure pos_weights is on same device as inputs
            if self.pos_weights.device != inputs.device:
                self.pos_weights = self.pos_weights.to(inputs.device)
            
            loss = F.binary_cross_entropy_with_logits(
                inputs, 
                targets, 
                pos_weight=self.pos_weights,
                reduction=self.reduction
            )
        else:
            # Standard BCE without weighting
            loss = F.binary_cross_entropy_with_logits(
                inputs, 
                targets,
                reduction=self.reduction
            )
        
        return loss


def calculate_focal_alpha(train_labels, task_type="multi_label"):
    """
    Calculate optimal focal loss alpha based on class distribution
    
    Alpha balances positive vs negative examples. Higher alpha = more weight on positive class.
    
    Formula:
        For multi-label with imbalanced data:
            - Calculate average positive frequency
            - Use sqrt transformation to boost low frequencies
            - This handles highly imbalanced data better (e.g., 5% pos -> alpha ≈ 0.22)
        
        For single-label: alpha = median class frequency
    
    Args:
        train_labels (ndarray): Training labels
                               Multi-label: shape (N, C) binary labels
                               Single-label: shape (N,) class indices
        task_type (str): "multi_label" or "single_label"
    
    Returns:
        float: Optimal alpha value in range [0.15, 0.9]
    
    Example:
        >>> # Multi-label with imbalanced data
        >>> train_labels = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
        >>> alpha = calculate_focal_alpha(train_labels, "multi_label")
        >>> print(f"Alpha: {alpha:.3f}")  # Alpha: ~0.29 (sqrt of ~8% positive rate)
        
        >>> # Single-label (3 classes)
        >>> train_labels = np.array([0, 1, 2, 0, 1])
        >>> alpha = calculate_focal_alpha(train_labels, "single_label")
        >>> print(f"Alpha: {alpha:.3f}")  # Alpha: 0.400
    """
    if task_type == "multi_label":
        # For multi-label: calculate average positive frequency across all labels
        num_positive = train_labels.sum(axis=0)  # Shape: (C,)
        total_per_label = len(train_labels)
        
        # Positive frequency per label
        pos_freq = num_positive / total_per_label
        
        # Average across all labels
        raw_alpha = pos_freq.mean()
        
        # For highly imbalanced data, use sqrt transformation to boost alpha
        # This prevents alpha from being too low
        # Example: 5% positive -> sqrt(0.05) ≈ 0.22
        #          10% positive -> sqrt(0.10) ≈ 0.32
        #          25% positive -> sqrt(0.25) = 0.50
        if raw_alpha < 0.15:
            alpha = np.sqrt(raw_alpha)
        else:
            alpha = raw_alpha
    
    else:  # single_label
        # For single-label: frequency of minority class(es)
        # Use frequency of non-majority class as alpha
        unique, counts = np.unique(train_labels, return_counts=True)
        total = len(train_labels)
        
        # Calculate frequency of each class
        class_freq = counts / total
        
        # Use median frequency as alpha (balances all classes)
        alpha = np.median(class_freq)
    
    # Clip to reasonable range [0.15, 0.9]
    # Minimum 0.15 to ensure positive class gets enough weight
    alpha = float(np.clip(alpha, 0.15, 0.9))
    
    return alpha


def calculate_pos_weights(train_labels):
    """
    Calculate positive weights for each label based on class frequency
    
    Formula:
        pos_weight[i] = num_negative[i] / num_positive[i]
    
    This gives higher weight to rare positive samples
    
    Args:
        train_labels (ndarray): Binary training labels, shape (N, C)
                               where N = num_samples, C = num_labels
    
    Returns:
        Tensor: Positive weights for each label, shape (C,)
    
    Example:
        >>> train_labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        >>> pos_weights = calculate_pos_weights(train_labels)
        >>> print(pos_weights)  # tensor([0.5, 0.5, 2.0])
    """
    # Count positive samples per label
    num_positive = train_labels.sum(axis=0)  # Shape: (C,)
    
    # Count negative samples per label
    num_negative = len(train_labels) - num_positive  # Shape: (C,)
    
    # Calculate weights (avoid division by zero)
    pos_weights = num_negative / (num_positive + 1e-7)
    
    # Clip extreme values to reasonable range
    # This prevents one label from dominating the loss
    pos_weights = np.clip(pos_weights, 0.1, 10.0)
    
    return torch.tensor(pos_weights, dtype=torch.float32)


# ============================================================================
# Custom Trainers with Different Loss Functions
# ============================================================================

class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer with Focal Loss
    
    Automatically uses Focal Loss instead of default BCEWithLogitsLoss
    
    Args:
        focal_alpha (float): Alpha parameter for Focal Loss
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


class WeightedBCETrainer(Trainer):
    """
    HuggingFace Trainer with Weighted BCE Loss
    
    Automatically uses Weighted BCE Loss instead of default BCEWithLogitsLoss
    
    Args:
        pos_weights (Tensor): Positive weights per label, shape (C,)
        *args, **kwargs: Arguments for base Trainer class
    
    Example:
        >>> pos_weights = calculate_pos_weights(train_labels)
        >>> trainer = WeightedBCETrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     pos_weights=pos_weights
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, *args, pos_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        self.weighted_bce = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use Weighted BCE Loss
        
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
        
        # Compute weighted BCE loss
        loss = self.weighted_bce(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
