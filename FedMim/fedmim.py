from models.base import BaseNet
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=in_features * 2)
        self.relu1 = nn.GELU(approximate='none')
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(in_features=in_features * 2, out_features=in_features)
        self.relu2 = nn.GELU(approximate='none')
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.dense1(x)))
        x = self.dropout2(self.relu2(self.dense2(x)))
        x = self.output_layer(x)
        return x


class FedMim(BaseNet):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(encoder,
                                   MLP(encoder.head.out_features, len(kwargs["classes"]))
                                   )
