import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FloorplanTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # CORRECTED LINE: Removed batch_first=True to match the forward pass logic
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]
            src_padding_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        # Note: input shape is [batch_size, seq_len]
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # PyTorch Transformer expects [seq_len, batch_size, embed_dim], so we permute
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        
        # TransformerDecoder needs memory, which for a decoder-only model is the src itself
        # The masks are now correctly aligned with the permuted src tensor
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=src_mask, tgt_key_padding_mask=src_padding_mask)
        
        # Permute back to [batch_size, seq_len, embed_dim]
        output = output.permute(1, 0, 2)
        output = self.output_layer(output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: str) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)