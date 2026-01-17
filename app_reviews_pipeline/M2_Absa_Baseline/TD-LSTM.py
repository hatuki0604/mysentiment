"""
python -m venv venv
source venv/bin/activate

pip install torch numpy pandas scikit-learn tqdm

pip install torch --index-url https://download.pytorch.org/whl/cu118

"""

def split_context(sentence, aspect):
    parts = sentence.lower().split(aspect.lower())
    if len(parts) == 1:
        # aspect không xuất hiện trong câu (rare case)
        return sentence, ""
    left = parts[0].strip()
    right = parts[1].strip()
    return left, right

import torch
import torch.nn as nn

class TDLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=150, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm_left = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm_right = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, left_seq, right_seq):
        left_embed = self.embedding(left_seq)
        right_embed = self.embedding(right_seq)

        _, (h_left, _) = self.lstm_left(left_embed)
        _, (h_right, _) = self.lstm_right(right_embed)

        h = torch.cat([h_left[-1], h_right[-1]], dim=1)
        out = self.fc(h)

        return out

