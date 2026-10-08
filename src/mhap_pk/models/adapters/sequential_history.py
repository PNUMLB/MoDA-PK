"""Causal residual history context for the native sequential PK observer."""
import torch
from torch import nn


class SequentialHistory(nn.Module):
    def __init__(self, mode, hidden):
        super().__init__()
        if mode not in ('summary', 'attention', 'raw_pool', 'raw_attention'):
            raise ValueError(mode)
        self.mode = mode
        if mode == 'summary':
            self.encoder = nn.Sequential(nn.Linear(7, hidden), nn.SiLU())
        elif mode.startswith('raw_'):
            # Observed log concentration, time/dose metadata; no model outputs.
            self.encoder = nn.Sequential(nn.Linear(11, hidden), nn.SiLU())
            if mode == 'raw_attention':
                self.attention = nn.MultiheadAttention(hidden, 2, batch_first=True)
        else:
            self.encoder = nn.Linear(12, hidden)
            self.attention = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.output = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, times, masks):
        x, t, valid = torch.stack(features, 1), torch.stack(times, 1), torch.stack(masks, 1)
        n = valid.sum(1).clamp_min(1)
        if self.mode.startswith('raw_'):
            age = (t[:, -1:] - t).clamp_min(0).log1p()
            clock = t.sign() * t.abs().log1p()
            raw = torch.cat([x[:, :, :1], x[:, :, 3:], clock[:, :, None], age[:, :, None]], -1)
            tokens = self.encoder(raw)
            if self.mode == 'raw_pool':
                h = (tokens * valid[:, :, None]).sum(1) / n[:, None]
            else:
                safe = valid.clone(); safe[:, 0] |= ~valid.any(1)
                h = self.attention(tokens[:, -1:], tokens, tokens,
                                   key_padding_mask=~safe, need_weights=False)[0][:, 0]
            return self.output(h) * valid.any(1)[:, None]
        residual = torch.where(valid, x[:, :, 2], 0.)
        first = torch.where(valid, t, torch.inf).min(1).values
        first = torch.where(valid.any(1), first, 0.)
        span = (t[:, -1] - first).clamp_min(0)
        if self.mode == 'summary':
            mean = residual.sum(1)/n
            first_index = valid.long().argmax(1)
            first_r = residual.gather(1, first_index[:, None]).squeeze(1)
            summary = torch.stack([mean, residual[:, -1], residual[:, -1]-first_r,
                (residual.square().sum(1)/n+1e-12).sqrt(), residual.abs().max(1).values,
                n.to(x.dtype).log1p(), span.log1p()], -1)
            h = self.encoder(summary)
        else:
            age = (t[:, -1:] - t).clamp_min(0).log1p()
            tokens = self.encoder(torch.cat([x, age[:, :, None]], -1))
            # All-masked profiles still need a finite attention computation.
            safe = valid.clone(); safe[:, 0] |= ~valid.any(1)
            h = self.attention(tokens[:, -1:], tokens, tokens,
                               key_padding_mask=~safe, need_weights=False)[0][:, 0]
        return self.output(h) * valid.any(1)[:, None]
