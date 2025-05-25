import torch
import torch.nn as nn

# Trained predictor for yield prediction
class NN(nn.Module):
    def __init__(self, in_dim, out_dim_class):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, in_dim//4),
            nn.ReLU(),
            nn.Linear(in_dim//4, out_dim_class)
        )

    def forward(self, x, c):
        xc = torch.cat((x, c), dim=1)
        y_pred = self.model(xc)
        y_pred = torch.clamp(y_pred, min=0.0, max=100.0)
        return y_pred
    
# Surrogate predictor for other tasks (with extra conditioning)
class NN_TASK(nn.Module):
    def __init__(self, in_dim, out_dim_class):
        super(NN_TASK, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, in_dim//4),
            nn.ReLU(),
            nn.Linear(in_dim//4, out_dim_class)
        )

    def forward(self, x, c, c_extra=None):
        if c_extra is not None:
            c = torch.cat((c, c_extra), dim=1)
        xc = torch.cat((x, c), dim=1)
        y_pred = self.model(xc)
        return y_pred