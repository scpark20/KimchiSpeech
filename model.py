import torch
from torch import nn
from torch.nn import functional as F
#import pytorch_lightning as pl

from STT import STTModel
from TTS import TTSModel

'''
Pytorch Model
'''
class Model(nn.Module):
    def __init__(self, stt_hparams, tts_hparams, mode='train'):
        super().__init__()
        self.hp = tts_hparams
        
        if mode == 'train':
            self.stt = STTModel(stt_hparams)
            self.global_step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
            
        self.tts = TTSModel(tts_hparams, mode=mode)
        
    def increase_step(self):
        self.global_step[0] = self.global_step + 1
        
    def _linear(self, start_value, end_value, current_index, start_index, end_index):
        if current_index > end_index:
            return end_value
        if current_index < start_index:
            return start_value

        grad = (end_value - start_value) / (end_index - start_index)
        y = start_value + grad * (current_index - start_index)
        
        return y.item()
        
    def forward(self, batch):
        stt_outputs = self.stt(batch)
        
        self.beta = self._linear(0, 1, self.global_step, 0, self.hp.annealing_steps)
        tts_outputs = self.tts(batch, stt_outputs, self.beta)
        
        return stt_outputs, tts_outputs
        
    def inference(self, cond, alignments=None, mel_length=None, temperature=1.0, speed=1.0, clip=None):
        y = self.tts.inference(cond, alignments, mel_length, temperature, speed, clip)
        
        return y

'''
Pytorch-Lightning Wrapping Model
'''
# class LightningModel(pl.LightningModule):
#     def __init__(self, stt_hparams, tts_hparams):
#         super().__init__()
#         self.hp = tts_hparams
#         self.model = Model(stt_hparams, tts_hparams)
#         self.save_hyperparameters()
        
#     def training_step(self, batch, batch_idx):
#         stt_outputs, tts_outputs = self.model(batch)
#         self.stt_outputs = stt_outputs
#         self.tts_outputs = tts_outputs
#         self.model.increase_step()
#         loss = stt_outputs['loss'] + tts_outputs['loss']
        
#         self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         self.log('stt', stt_outputs['loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         self.log('recon', tts_outputs['recon_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         self.log('kl', tts_outputs['kl_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         self.log('param', tts_outputs['param_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
#         self.log('beta', self.model.beta, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
#         return loss
        
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
#         return optimizer
