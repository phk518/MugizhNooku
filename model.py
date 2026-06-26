import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class DigitalTwinPredictor(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super(DigitalTwinPredictor, self).__init__()
        self.convlstm = ConvLSTMCell(input_dim=input_channels, hidden_dim=hidden_channels)
        self.final_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        # Input tensor dimension format: (Batch, Sequence, Channels, Height, Width)
        b, seq_len, _, h, w = x.size()
        h_t = torch.zeros(b, self.convlstm.hidden_dim, h, w).to(x.device)
        c_t = torch.zeros(b, self.convlstm.hidden_dim, h, w).to(x.device)
        
        for t in range(seq_len):
            h_t, c_t = self.convlstm(x[:, t, :, :, :], (h_t, c_t))
            
        return self.final_conv(h_t)
