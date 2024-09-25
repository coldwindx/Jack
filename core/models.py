import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

@torch.jit.export
def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param inputs_words: [Batch, SeqLen, input_dim]
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

class DeepRan(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_layers, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=input_dim)
        # BiLstm
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        # Output classifier per sequence lement
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.randn(num_layers * 2, model_dim), requires_grad=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(model_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        text, sorted_seq_lengths, _ = prepare_pack_padded_sequence(x, lengths)
        embed = self.bn(text.transpose(1, 2)).transpose(1, 2)
        sorted_seq_lengths = sorted_seq_lengths.cpu()
        packed_embedded = pack_padded_sequence(embed, sorted_seq_lengths, batch_first=True)
        _, (hidden, _) = self.lstm(packed_embedded)

        alpha = torch.sum(torch.mul(self.w, hidden.permute(1, 0, 2)), dim=1)
        logits = self.fc(alpha)
        return logits


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        return z, h
