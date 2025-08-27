import torch.nn as nn
import torch

class FusionModel(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=1024, out_dim=126):
        super().__init__()
        self.fc_text = nn.Linear(text_dim, hidden_dim)
        self.fc_img = nn.Linear(img_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, text_feat, img_feat):
        x = torch.stack([self.fc_text(text_feat), self.fc_img(img_feat)], dim=1)
        attn_out, _ = self.attn(x, x, x)
        return self.mlp(attn_out.mean(dim=1))
