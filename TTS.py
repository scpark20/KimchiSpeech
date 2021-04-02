import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import time
from TruncatedNormal import TruncatedStandardNormal

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, 
                 zero_weight=False, weight_norm=False):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if zero_weight:
            self.conv.weight.data.zero_()
        else:
            self.conv.weight.data.normal_(0, 0.02)
            
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
    
    def forward(self, x):
        # x : (b, c, t)
        x = self.conv(x)
        
        return x

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        x = self.conv(x)

        return x
    
class TTSTextEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.convs = nn.ModuleList()
        for i in range(hp.text_encoder_n_convs):
            conv = nn.Sequential(nn.Conv1d(hp.text_encoder_dim if i > 0 else hp.dec_dim,
                                           hp.text_encoder_dim,
                                           kernel_size=hp.text_encoder_kernel_size, 
                                           padding=(hp.text_encoder_kernel_size-1)//2),
                                 nn.BatchNorm1d(hp.text_encoder_dim))
            self.convs.append(conv)
            
        self.lstm  = nn.LSTM(hp.text_encoder_dim, 
                             hp.text_encoder_dim//2,
                             batch_first=True,
                             bidirectional=True)
        
        if hp.enc_add:
            self.enc_linear = nn.Linear(hp.text_encoder_dim, hp.dec_dim)
            
        if self.hp.attention == 'Gaussian':
            num_params = 2 
        elif self.hp.attention == 'Laplace':
            num_params = 3
            
        self.param_linear = nn.Linear(hp.text_encoder_dim, num_params)
        self.param_linear.weight.data.zero_()
        
    def forward(self, x, input_lengths):
        # x : (b, c, l)
        
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        
        # (b, l, c)
        x = x.transpose(1, 2)
        
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        # (b, l, c)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # (b, l, 2)
        params = self.param_linear(x)
        
        if self.hp.enc_add:
            encoded = self.enc_linear(x)
            return encoded, params
        else:
            return params
    
    def inference(self, x, time_dict):
        
        t0 = time.time()
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        t1 = time.time()
        time_dict['enc_conv'] = t1 - t0
        
        # (b, l, c)
        x = x.transpose(1, 2)

        t0 = time.time()
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        t1 = time.time()
        time_dict['enc_lstm'] = t1 - t0
        
        # (b, l, 2)
        params = self.param_linear(x)
        
        if self.hp.enc_add:
            encoded = self.enc_linear(x)
            return encoded, params
        else:
            return params
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = x * self.sigmoid(x)
        
        return y
    
class SELayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(Conv1d(in_channels, in_channels),
                                  nn.Sigmoid())
        
    def forward(self, x):
        # x : (b, c, t)
        y = x.mean(dim=2, keepdim=True)
        y = self.conv(y)
        y = x * y
        
        return y
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, last_zero=False, type=0):
        super().__init__()
            
        if type == 0:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels)
                                     )
        if type == 1:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels)
                                     )    
        if type == 2:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
                                     )        
        if type == 3:
            self.conv = nn.Sequential(nn.BatchNorm1d(in_channels),
                                      nn.GELU(),
                                      Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        
        if type == 4:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels, weight_norm=True),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, weight_norm=True),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, weight_norm=True),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels, weight_norm=True)
                                     )    
        if type == 5:
            self.conv = nn.Sequential(nn.BatchNorm1d(in_channels),
                                      Conv1d(in_channels, hidden_channels),
                                      nn.BatchNorm1d(hidden_channels),
                                      Swish(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                                      nn.BatchNorm1d(hidden_channels),
                                      Swish(),
                                      Conv1d(hidden_channels, out_channels),
                                      #nn.BatchNorm1d(out_channels),
                                      #SELayer(out_channels)
                                     )
            
        if type == 6:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels)
                                     )  
            
        if type == 7:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels),
                                      nn.BatchNorm1d(hidden_channels),
                                      Swish(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      Swish(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      Swish(),
                                      Conv1d(hidden_channels, out_channels)
                                     )    
            
        if type == 8:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels)
                                     )
            
        if type == 9:
            self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(hidden_channels),
                                      nn.GELU(),
                                      Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
                                     )
        
    def forward(self, x):
        x = self.conv(x)
        
        return x
    
class TTSMelEncoderBlocks(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        self.convs = nn.ModuleList([ConvBlock(hp.enc_dim, hp.enc_hidden_dim, hp.enc_dim, type=hp.conv_type) \
                                    for _ in range(hp.n_blocks)])
        
    def forward(self, x):
        xs = []
        for conv in self.convs:
            if self.hp.encoder_residual:
                x = x + conv(x)
            else:
                x = conv(x)
            xs.append(x)
        xs.reverse()
        
        return x, xs
    
class TTSMelEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        self.in_layer = Conv1d(hp.n_mels, hp.enc_dim)
        self.encoder_blocks = nn.ModuleList([TTSMelEncoderBlocks(hp) for _ in range(hp.n_layers)])
        self.downs = nn.ModuleList([Conv1d(hp.enc_dim, hp.enc_dim, kernel_size=2, stride=2) for _ in range(hp.n_layers)])
        
    def forward(self, x):
        # x : (b, c, t)
        
        x = self.in_layer(x)
        xs_list = []
        for block, down in zip(self.encoder_blocks, self.downs):
            x, xs = block(x)
            xs_list.append(xs)
            x = down(x)
        xs_list.reverse()
        
        return xs_list
    
class TTSMelDecoderBlock(nn.Module):
    def __init__(self, hp, layer, mode):
        super().__init__()
        
        if hp.decoder_expand_dim:
            dec_dim = hp.dec_dim//(2**(hp.n_layers-layer-1))
            dec_hidden_dim = hp.dec_hidden_dim//(2**(hp.n_layers-layer-1))
        else:
            dec_dim = hp.dec_dim
            dec_hidden_dim = hp.dec_hidden_dim
        
        self.hp = hp
        if mode == 'train':
            self.q = ConvBlock(dec_dim + hp.enc_dim, dec_hidden_dim, hp.z_dim*2, last_zero=True, type=hp.conv_type)
        self.clip = hp.truncated_max
        self.tsn = TruncatedStandardNormal(a=-self.clip, b=self.clip)
        if hp.z_proj:
            self.z_proj = Conv1d(hp.z_dim, dec_dim)
        self.out = ConvBlock(dec_dim, dec_hidden_dim, dec_dim, type=hp.conv_type)
        
    def _get_kl_div(self, q_params):
        p_mean = 0
        p_logstd = 0
        q_mean = q_params[0]
        q_logstd = q_params[1]
        
        return -q_logstd + 0.5 * (q_logstd.exp() ** 2 + q_mean ** 2) - 0.5
    
    def _sample_from_q(self, q_params):
        mean = q_params[0]
        logstd = q_params[1]
        sample = mean + mean.new(mean.shape).normal_() * logstd.exp()
        
        return sample
    
    def _sample_from_p(self, tensor, shape, temperature=1.0, clip=None):
        if clip is None:
            #print('Normal')
            sample = tensor.new(*shape).normal_() * temperature
        else:
            #print('TruncatedNormal', clip)
            tsn = TruncatedStandardNormal(a=-clip, b=clip)
            sample = tsn.rsample(tensor, shape)
            sample = sample * temperature
        
        return sample
    
    def forward(self, x, src, cond):
        if x is None:
            y = x = cond
        else:
            y = x + cond
        
        q_params = self.q(torch.cat([y, src], dim=1)).split(self.hp.z_dim, dim=1)
        kl_div = self._get_kl_div(q_params)
        z = self._sample_from_q(q_params)
        
        if self.hp.z_proj:
            y = y + self.z_proj(z)
        else:
            y[:, :self.hp.z_dim] = y[:, :self.hp.z_dim] + z
        
        if self.hp.decoder_residual:
            y = x + self.out(y)
        else:
            y = self.out(y)
        
        return y, kl_div
    
    def inference(self, x, cond, temperature, clip, time_dict):
        if x is None:
            y = x = cond
        else:
            y = x + cond

        t0 = time.time()
        z = self._sample_from_p(x, (x.size(0), self.hp.z_dim, x.size(2)), temperature, clip)
        t1 = time.time()
        time_dict['random'] = time_dict['random'] + (t1 - t0)
        
        t0 = time.time()
        
        if self.hp.z_proj:
            y = y + self.z_proj(z)
        else:
            y[:, :self.hp.z_dim] = y[:, :self.hp.z_dim] + z
        t1 = time.time()
        time_dict['z_proj'] = time_dict['z_proj'] + (t1 - t0)
        
        t0 = time.time()
        if self.hp.decoder_residual:
            y = x + self.out(y)
        else:
            y = self.out(y)
        t1 = time.time()
        time_dict['last_conv'] = time_dict['last_conv'] + (t1 - t0)
        
        return y
        
class TTSMelDecoderBlocks(nn.Module):
    def __init__(self, hp, layer, mode):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([TTSMelDecoderBlock(hp, layer, mode) for _ in range(hp.n_blocks)])
        
    def forward(self, x, srcs, cond):
        
        kl_divs = []
        for decoder_block, src in zip(self.decoder_blocks, srcs):
            x, kl_div = decoder_block(x, src, cond)
            kl_divs.append(kl_div)
            
        return x, kl_divs
    
    def inference(self, x, cond, temperature, clip, time_dict):
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block.inference(x, cond, temperature, clip, time_dict)
            
        return x

class TTSMelDecoder(nn.Module):
    def __init__(self, hp, mode):
        super().__init__()
        self.decoder_blocks_list = nn.ModuleList([TTSMelDecoderBlocks(hp, layer, mode) for layer in range(hp.n_layers)])
        if hp.decoder_expand_dim:
            ups = []
            for i in range(hp.n_layers-1):
                up = ConvTranspose1d(hp.dec_dim//(2**(i+1)), hp.dec_dim//(2**i), kernel_size=2, stride=2)
                ups.append(up)
            ups.append(nn.Identity())
            ups.reverse()
            self.ups = nn.ModuleList(ups)
        else:
            self.ups = nn.ModuleList([nn.Identity() if i == 0 else \
                                      ConvTranspose1d(hp.dec_dim, hp.dec_dim, kernel_size=2, stride=2) for i in range(hp.n_layers)
                                     ])
            
        self.out = Conv1d(hp.dec_dim, hp.n_mels, kernel_size=3, padding=1)
        
    def forward(self, srcs, conds):
        x = None
        kl_divs = []
        
        for decoder_blocks, up, src, cond in zip(self.decoder_blocks_list, self.ups, srcs, conds):
            if x is not None:
                x = up(x)
            x, kl_div = decoder_blocks(x, src, cond)
            kl_divs.extend(kl_div)
        x = self.out(x)
        
        return x, kl_divs
        
    def inference(self, conds, temperature, clip, time_dict):
        x = None
        for decoder_blocks, up, cond in zip(self.decoder_blocks_list, self.ups, conds):
            t0 = time.time()
            if x is not None:
                x = up(x)
            t1 = time.time()
            time_dict['up'] = time_dict['up'] + (t1 - t0)
            x = decoder_blocks.inference(x, cond, temperature, clip, time_dict)
        x = self.out(x)
        
        return x
            
class Pooling(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.decoder_expand_dim:
            self.poolings = nn.ModuleList([Conv1d(hp.dec_dim//(2**i), hp.dec_dim//(2**(i+1)), kernel_size=2, stride=2) \
                                           for i in range(hp.n_layers-1)])
        else:
            self.poolings = nn.ModuleList([Conv1d(hp.dec_dim, hp.dec_dim, kernel_size=2, stride=2) for _ in range(hp.n_layers-1)])
    
    def forward(self, x):
        xs = [x]
        for pooling in self.poolings:
            x = pooling(x)
            xs.append(x)
        xs.reverse()
        
        return xs
        
class TTSModel(nn.Module):
    def __init__(self, hp, mode='train'):
        super().__init__()
        
        self.hp = hp
        self.length_unit = 2 ** (hp.n_layers-1)
        
        self.embedding = nn.Embedding(hp.n_symbols, hp.dec_dim)
        self.text_encoder = TTSTextEncoder(hp)
        self.pooling = Pooling(hp)
        if mode == 'train':
            self.mel_encoder = TTSMelEncoder(hp)
        self.mel_decoder = TTSMelDecoder(hp, mode)
        
    def _get_loss(self, src, pred, kl_divs, stt_params, tts_params, beta):
        # Reconstruction Loss
        recon_loss = ((pred - src) ** 2).sum(dim=[1, 2])
        
        # KL-Divergence Loss
        kl_loss = None
        for kl in kl_divs:
            kl_loss = kl.sum(dim=[1, 2]) if kl_loss is None else kl_loss + kl.sum(dim=[1, 2])
        dim = src.size(1) * src.size(2)
        
        # Alignment Parameters Loss
        param_loss = ((stt_params - tts_params) ** 2).mean()
        
        # Loss
        loss = (recon_loss + beta * kl_loss).mean() / dim + param_loss
        recon_loss = recon_loss.mean() / dim
        kl_loss = kl_loss.mean() / dim
        
        return loss, recon_loss, kl_loss, param_loss
    
    def _normalize(self, alignments, eps=1e-8):
        # alignments : (b, l, t)
        alignments = (alignments + eps).log().softmax(dim=1)
        
        return alignments
    
    def _adjust_mean(self, params, text_lengths, mel_lengths):
        # params : (b, l, 2)
        # lengths : (b)
        
        mean = (params[:, :, 0].exp() * self.hp.mean_coeff).cumsum(dim=1)
        
        for i, (text_length, mel_length) in enumerate(zip(text_lengths, mel_lengths)):
            e = mel_length - mean[i, text_length-2]
            #print(i, mel_length, mean[i, text_length-2])
            e = e / self.hp.mean_coeff
            #print(i, e)
            e = torch.clamp(e, min=1e-8)
            #print(i, e)
            end_length = torch.log(e)
            #print(i, end_length)
            params[i, text_length-1, 0] = end_length
            params[i, text_length-1, 1] = np.log(1e-8)
        
        print(params[0, text_lengths[0]-1, 0])
        return params
    
    def _get_attention_matrix(self, hp, params, mel_length, speed=1.0):
        if hp.attention == 'Gaussian':
            batch, text_length, _ = params.size()
            
            delta_mean = params[:, :, 0:1].exp() * self.hp.mean_coeff * speed
            #print(delta_mean)
            mean = delta_mean.cumsum(dim=1)
            #print(mean)
            if mel_length is None:
                mel_length = torch.max(mean).long().item()
            scale = params[:, :, 1:2].exp() * self.hp.scale_coeff / speed
            Z = torch.sqrt(2 * np.pi * scale ** 2)
            matrix = torch.linspace(0, mel_length-1, mel_length, device=params.device).repeat(batch, text_length, 1)
            matrix = 1 / Z * torch.exp(-0.5 * (matrix - mean) ** 2 / (scale ** 2))
        
        elif hp.attention == 'Laplace':
            batch, text_length, _ = params.size()
    
            mean = (params[:, :, 0:1].exp() * self.hp.mean_coeff * speed).cumsum(dim=1)
            if mel_length is None:
                mel_length = torch.max(mean).long().item()
            scale = params[:, :, 1:2].exp() * self.hp.scale_coeff / speed # 0.2
            asym = params[:, :, 2:3].exp()
            Z = scale / (asym + 1 / asym)
            matrix = torch.linspace(0, mel_length-1, mel_length, device=params.device).repeat(batch, text_length, 1)
            p = matrix - mean
            p_pos = -scale * asym * p
            p_neg = scale / asym * p
            matrix = Z * torch.exp(p_pos * (p>=0) + p_neg * (p<0))
        
        return matrix
            
    def forward(self, batch, stt_outputs, beta=1.0):
        
        # (b, c, t)
        x = batch['mels']
        # (b, l)
        cond = batch['text']
        
        stt_params = stt_outputs['alignment_params'].detach()
        stt_params = self._adjust_mean(stt_params, batch['text_lengths'], batch['mel_lengths'])
        stt_alignments = self._get_attention_matrix(self.hp, stt_params, torch.max(batch['mel_lengths']).item())
        stt_alignments = self._normalize(stt_alignments)
        
        # Pad
        pad_length = ((x.size(2)-1)//self.length_unit+1) * self.length_unit-x.size(2)
        x = F.pad(x, (0, pad_length), value=0 if self.hp.mel_norm else -5)
        stt_alignments = F.pad(stt_alignments, (0, pad_length))
        
        # Get Condition
        # (b, c, l)
        cond = self.embedding(cond).transpose(1, 2)
        # (b, l, 2)
        if self.hp.enc_add:
            encoded, params = self.text_encoder(cond, batch['text_lengths'])
            cond = cond + encoded.transpose(1, 2)
        else:
            params = self.text_encoder(cond, batch['text_lengths'])
        # (b, c, t)
        alignments = self._normalize(self._get_attention_matrix(self.hp, params, x.size(2)))
        # (b, c, t)
        cond = torch.bmm(cond, stt_alignments)
        
        # Get Pooled Conditions and Sources
        # [(b, c, t)...]
        conds = self.pooling(cond)
        # [(b, c, t)...]
        xs = self.mel_encoder(x)
        
        # Get Prediction and KL-div.
        y, kl_divs = self.mel_decoder(xs, conds)
        
        loss, recon_loss, kl_loss, param_loss = self._get_loss(x, y, kl_divs, stt_params, params, beta=beta)
        
        outputs = {'mels': x,
                   'pred': y,
                   'loss': loss,
                   'recon_loss': recon_loss,
                   'kl_loss': kl_loss,
                   'param_loss': param_loss,
                   'stt_alignments': stt_alignments,
                   'alignments': alignments}
        
        return outputs
        
    def inference(self, cond, mel_length=None, alignments=None, temperature=1.0, speed=1.0, clip=None):
        # cond : (b, l)
        
        time_dict = {'alignment': 0.0,
                     'random': 0.0,
                     'last_conv': 0.0, 
                     'z_proj': 0.0,
                     'cond': 0.0,
                     'up': 0.0,
                     'encode': 0.0,
                     'pad': 0.0,
                     'enc_conv': 0.0,
                     'enc_lstm': 0.0,
                    }
        
        t0 = time.time()
        # (b, c, l)
        cond = self.embedding(cond).transpose(1, 2)
        # (b, l, 2)
        if self.hp.enc_add:
            encoded, params = self.text_encoder.inference(cond, time_dict)
            cond = cond + encoded.transpose(1, 2)
        else:
            params = self.text_encoder.inference(cond, time_dict)
        t1 = time.time()
        time_dict['encode'] = t1 - t0
        
        t0 = time.time()
        if alignments is None:
            alignments = self._get_attention_matrix(self.hp, params, mel_length, speed)
        alignments = self._normalize(alignments)
        t1 = time.time()
        time_dict['alignment'] = t1 - t0
        
        t0 = time.time()
        # Pad
        pad_length = ((alignments.size(2)-1)//self.length_unit+1) * self.length_unit-alignments.size(2)
        alignments = F.pad(alignments, (0, pad_length))
        t1 = time.time()
        time_dict['pad'] = t1 - t0
        
        t0 = time.time()
        # (b, c, t)
        cond = torch.bmm(cond, alignments)
        # [(b, c, t)...]
        conds = self.pooling(cond)
        t1 = time.time()
        time_dict['cond'] = t1 - t0
        
        y = self.mel_decoder.inference(conds, temperature, clip, time_dict)
        if pad_length > 0:
            y = y[:, :, :-pad_length]
        
        return y, time_dict