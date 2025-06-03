import torch
import torch.nn as nn

class DouZeroNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(state_dim)
        
        self.gru = nn.GRU(128, 128, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size()[1:])))
    
    def forward(self, x):
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, 128, -1).transpose(1, 2)
        
        # GRU处理序列特征
        gru_out, _ = self.gru(conv_out)
        gru_out = gru_out[:, -1, :]
        
        return self.fc(gru_out)