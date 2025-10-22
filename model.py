import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class PhoBERTForABSA(nn.Module):
    """PhoBERT model for Aspect-Based Sentiment Analysis"""
    
    def __init__(self, pretrained_model: str, num_labels: int, dropout: float = 0.1):
        super(PhoBERTForABSA, self).__init__()
        
        self.num_labels = num_labels
        self.phobert = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.phobert.config.hidden_size
        
        # Classifier head
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length) - not used for RoBERTa
            labels: (batch_size,) - optional, for computing loss
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        output = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            output['loss'] = loss
        
        return output
    
    def save_pretrained(self, save_path):
        """Save model"""
        self.phobert.save_pretrained(save_path)
        torch.save({
            'num_labels': self.num_labels,
            'classifier_state_dict': self.classifier.state_dict(),
            'dropout_p': self.dropout.p
        }, f"{save_path}/classifier_head.pt")
    
    @classmethod
    def from_pretrained(cls, save_path, pretrained_model=None):
        """Load model"""
        classifier_info = torch.load(f"{save_path}/classifier_head.pt")
        
        if pretrained_model is None:
            pretrained_model = save_path
        
        model = cls(
            pretrained_model=pretrained_model,
            num_labels=classifier_info['num_labels'],
            dropout=classifier_info['dropout_p']
        )
        
        model.classifier.load_state_dict(classifier_info['classifier_state_dict'])
        
        return model
