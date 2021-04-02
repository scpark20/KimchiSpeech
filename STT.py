import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class STTPrenet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.layers = nn.Sequential(nn.Linear(hp.embedding_dim, hp.prenet_dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(hp.prenet_dim, hp.prenet_dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        
    def forward(self, x):
        x = self.layers(x)
        
        return x
    
class GaussianAttention(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.param_linear = nn.Linear(hp.attention_rnn_dim, 2)
        self.param_linear.weight.data.zero_()
        
    def init_mean(self, tensor, batch_size):
        self.mean = tensor.data.new(batch_size, 1).zero_()
        
    def _linspace(self, tensor, batch, length):
        # (l)
        lin = torch.linspace(start=0, end=length-1, steps=length, device=tensor.device)
        # (b, l)
        lin = lin.unsqueeze(0).repeat(batch, 1)
        
        return lin
    
    def _get_weight(self, params, length):
        
        # Mean
        mean_delta = torch.exp(params[:, 0:1]) * self.hp.mean_coeff
        self.mean = self.mean + mean_delta
        
        # Scale
        scale = torch.exp(params[:, 1:2]) * self.hp.scale_coeff
        
        # Z
        Z = torch.sqrt(2 * np.pi * scale ** 2)
        
        # (b, l)
        lin = self._linspace(params, batch=params.size(0), length=length)
        # (b, l)
        weight = 1 / Z * torch.exp(-0.5 * (lin - self.mean) ** 2 / (scale ** 2))
        
        return weight
        
    def forward(self, attention_hidden, memory, mask):
        
        # (b, 2)
        params = self.param_linear(attention_hidden)
        # (b, l)
        weight = self._get_weight(params, memory.size(1))
        if mask is not None:
            weight.data.masked_fill_(mask, 0)
        
        # (b, 1, c)
        context = torch.bmm(weight.unsqueeze(1), memory)
        # (b, c)
        context = context.squeeze(1)
        
        return context, weight, params
    
class LaplaceAttention(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.param_linear = nn.Linear(hp.attention_rnn_dim, 3)
        self.param_linear.weight.data.zero_()
        
    def init_mean(self, tensor, batch_size):
        self.mean = tensor.data.new(batch_size, 1).zero_()
        
    def _linspace(self, tensor, batch, length):
        # (l)
        lin = torch.linspace(start=0, end=length-1, steps=length, device=tensor.device)
        # (b, l)
        lin = lin.unsqueeze(0).repeat(batch, 1)
        
        return lin
    
    def _get_weight(self, params, length):
        
        # Mean
        mean_delta = torch.exp(params[:, 0:1]) * self.hp.mean_coeff
        self.mean = self.mean + mean_delta
        
        # Scale
        scale = torch.exp(params[:, 1:2]) * self.hp.scale_coeff
        
        # Asymmetry
        asym = torch.exp(params[:, 2:3])
        
        # Z
        Z = scale / (asym + 1 / asym)
        
        # (b, l)
        lin = self._linspace(params, batch=params.size(0), length=length)
        p = lin - self.mean
        p_pos = -scale * asym * p
        p_neg = scale / asym * p
        
        weight = Z * torch.exp(p_pos * (p>=0) + p_neg * (p<0))
        
        return weight
        
    def forward(self, attention_hidden, memory, mask):
        
        # (b, 2)
        params = self.param_linear(attention_hidden)
        # (b, l)
        weight = self._get_weight(params, memory.size(1))
        if mask is not None:
            weight.data.masked_fill_(mask, 0)
        
        # (b, 1, c)
        context = torch.bmm(weight.unsqueeze(1), memory)
        # (b, c)
        context = context.squeeze(1)
        
        return context, weight, params

        
class STTEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for i in range(hp.encoder_n_convs):
            conv = nn.Sequential(nn.Conv1d(hp.encoding_dim if i > 0 else hp.n_mels,
                                           hp.encoding_dim,
                                           kernel_size=hp.encoder_kernel_size, 
                                           padding=(hp.encoder_kernel_size-1)//2),
                                 nn.BatchNorm1d(hp.encoding_dim))
            self.convs.append(conv)
            
        self.lstm  = nn.LSTM(hp.encoding_dim, 
                             hp.encoding_dim//2,
                             batch_first=True,
                             bidirectional=True)
        
    def forward(self, mels, input_lengths):
        # mels : (b, c, t)
        
        x = mels
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        
        # (b, t, c)
        x = x.transpose(1, 2)
        
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        # (b, t, c)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        return x
    
class STTDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.prenet = STTPrenet(hp)
        self.attention_rnn = nn.LSTMCell(hp.prenet_dim + hp.encoding_dim, hp.attention_rnn_dim)
        if hp.attention == 'Gaussian':
            self.attention = GaussianAttention(hp)
        elif hp.attention == 'Laplace':
            self.attention = LaplaceAttention(hp)
            
        self.decoder_rnn = nn.LSTMCell(hp.attention_rnn_dim + hp.encoding_dim, hp.decoder_rnn_dim)
        self.output_linear = nn.Linear(hp.decoder_rnn_dim + hp.encoding_dim, hp.n_symbols)
        
    def _get_go_frame(self, memory):
        
        decoder_input = memory.data.new(memory.size(0), self.hp.embedding_dim).zero_()
        return decoder_input
    
    def _init_decoder_states(self, memory, mask):
        
        b, l, _ = memory.size()
        
        self.attention_hidden = memory.data.new(b, self.hp.attention_rnn_dim).zero_()
        self.attention_cell = memory.data.new(b, self.hp.attention_rnn_dim).zero_()
        
        self.decoder_hidden = memory.data.new(b, self.hp.decoder_rnn_dim).zero_()
        self.decoder_cell = memory.data.new(b, self.hp.decoder_rnn_dim).zero_()
        
        self.attention_context = memory.data.new(b, self.hp.encoding_dim).zero_()
        
        self.memory = memory
        self.mask = mask
        
    def _get_mask_from_lengths(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        
        return mask
    
    def _decode(self, decoder_input):
        # decoder_input : (b, c)
        
        # Attention RNN
        cell_input = torch.cat([decoder_input, self.attention_context], dim=1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.hp.p_attention_dropout, self.training)
        
        # Attention
        self.attention_context, weight, params = self.attention(self.attention_hidden, self.memory, self.mask)
        
        # Decoder
        decoder_input = torch.cat([self.attention_hidden, self.attention_context], dim=1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.hp.p_decoder_dropout, self.training)
        
        # Output Linear
        decoder_hidden_attention_context = torch.cat([self.decoder_hidden, self.attention_context], dim=1)
        decoder_output = self.output_linear(decoder_hidden_attention_context)
        
        return decoder_output, weight, params
        
    def forward(self, memory, decoder_inputs, memory_lengths):
        # memory : ()
        # decoder_inputs : (b, c, l)
        # memory_lengths : (b)
        
        # (1, b, c)
        decoder_input = self._get_go_frame(memory).unsqueeze(0)
        # (l, b, c)
        decoder_inputs = decoder_inputs.permute(2, 0, 1)
        # (1+l, b, c)
        decoder_inputs = torch.cat([decoder_input, decoder_inputs], dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        
        self._init_decoder_states(memory, mask=~self._get_mask_from_lengths(memory_lengths))
        self.attention.init_mean(memory, memory.size(0))
        
        logits, alignments, params = [], [], []
        while len(logits) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(logits)]
            logit, alignment, param = self._decode(decoder_input)
            logits.append(logit)
            alignments.append(alignment)
            params.append(param)
        
        logits = torch.stack(logits, dim=1)
        alignments = torch.stack(alignments, dim=1)
        params = torch.stack(params, dim=1)
        
        return logits, alignments, params
            
class STTModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.embedding = nn.Embedding(hp.n_symbols, hp.embedding_dim)
        self.encoder = STTEncoder(hp)
        self.decoder = STTDecoder(hp)

    def forward(self, batch):
        if self.hp.mel_norm:
            mels = batch['mels']
        else:
            mels = (batch['mels'] + 5) / 5
            
        # (b, c, t)
        encoded_inputs = self.encoder(mels, batch['mel_lengths'])
        # (b, c, l)
        embedded_outputs = self.embedding(batch['text']).transpose(1, 2)
        # logits : (b, c, l), alignments : (b, l, t), alignment_params : (b, l, c)
        logits, alignments, alignment_params = self.decoder(encoded_inputs, embedded_outputs, batch['mel_lengths'])
        loss = nn.CrossEntropyLoss()(logits.transpose(1, 2), batch['text'])
        
        outputs = {'loss': loss,
                   'mels': mels,
                   'alignments': alignments,
                   'alignment_params': alignment_params
                  }
        
        return outputs