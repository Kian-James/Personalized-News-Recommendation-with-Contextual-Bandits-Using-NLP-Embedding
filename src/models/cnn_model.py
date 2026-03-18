import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes, kernel_sizes=[2,3,4], num_filters=100):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, L_out)
        pooled = [F.max_pool1d(out, kernel_size=out.shape[2]).squeeze(2) for out in conv_outs]  # (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        return self.fc(cat)