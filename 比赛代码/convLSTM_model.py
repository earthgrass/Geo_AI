"""
纯净版模型库 (convLSTM_model.py)
职责: 仅定义网络架构，提供给训练(Step 2.2)和预测(Step 2.3)统一调用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        
        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, *image_size, device=device),
                torch.zeros(batch_size, self.hidden_dim, *image_size, device=device))

class SpatialResidualConvLSTM(nn.Module):
    def __init__(self, input_channels=4, hidden_dims=[64, 128], kernel_size=3):
        super(SpatialResidualConvLSTM, self).__init__()
        self.encoder1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        self.encoder2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.pred_head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)
        
        self.residual_net = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        b, seq_len, _, h, w = x.size()
        h1, c1 = self.encoder1.init_hidden(b, (h, w))
        h2, c2 = self.encoder2.init_hidden(b, (h, w))
        
        for t in range(seq_len):
            h1, c1 = self.encoder1(x[:, t, :, :, :], (h1, c1))
            h2, c2 = self.encoder2(h1, (h2, c2))
            
        dec_feat = self.decoder(h2)
        p_pred = self.pred_head(dec_feat)
        
        res_input = torch.cat([h2, p_pred], dim=1) 
        delta_p = self.residual_net(res_input)
        return F.relu(p_pred + delta_p)