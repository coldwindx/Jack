import os
import sys
from typing import Optional
from torch import optim
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy

import lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

from .transformer import SingleChannelTransformer
from .network import CosineWarmupScheduler

class SingleChannelPredictor(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = SingleChannelTransformer(vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout, input_dropout)

    def forward(self, x, mask: Optional [torch.Tensor]=None, add_positional_encoding: bool=True):
         return self.net(x, mask=mask, add_positional_encoding=add_positional_encoding)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, epochs=self.hparams.max_iters)
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        self.log("learning_rate", optimizer.param_groups[0]['lr'], on_step=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("train_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("valid_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("valid_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("test_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)
        return preds


class SingleChannelClassifier(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = SingleChannelTransformer(vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout, input_dropout)

    def forward(self, x, mask=None, add_positional_encoding=True):
         return self.net(x, mask=mask, add_positional_encoding=add_positional_encoding)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, epochs=self.hparams.max_iters)
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        self.log("learning_rate", optimizer.param_groups[0]['lr'], on_step=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("train_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("valid_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("valid_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("test_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        inp_data, mask, _, labels = batch
        preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)
        preds = preds.reshape(labels.shape)
        return preds

