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
from .models import DeepRan

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
    def __init__(self, transformer, lr, warmup, max_iters, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = transformer

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

class DeepRanPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_layers, lr, warmup, max_iters, dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = DeepRan(input_dim, model_dim, num_classes, num_layers, dropout)

    def forward(self, tfidfs, vecs, lengths=None):
        return self.net(tfidfs.unsqueeze(2) * vecs, lengths)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, epochs=self.hparams.max_iters)
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        self.log("learning_rate", optimizer.param_groups[0]['lr'], on_step=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        tfidfs, vecs, lengths, labels = batch
        preds = self.forward(tfidfs, vecs, lengths=lengths)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("train_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tfidfs, vecs, lengths, labels = batch
        preds = self.forward(tfidfs, vecs, lengths=lengths)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("valid_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("valid_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        tfidfs, vecs, lengths, labels = batch
        preds = self.forward(tfidfs, vecs, lengths=lengths)
        preds = preds.reshape(labels.shape)

        loss = F.binary_cross_entropy(preds, labels)
        acc = binary_accuracy(preds, labels, threshold=0.5)

        self.log("test_loss", loss, on_epoch=True, enable_graph=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, enable_graph=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        tfidfs, vecs, lengths, labels = batch
        preds = self.forward(tfidfs, vecs, lengths=lengths)
        preds = preds.reshape(labels.shape)
        return preds

from torch import nn
from .models import AutoEncoder
class DeepGuard(pl.LightningModule):
    def __init__(self, input_dim, model_dim, output_dim, lr, warmup, max_iters, dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.batchlayer = nn.BatchNorm1d(num_features=input_dim)
        self.autoencoder = AutoEncoder(self.hparams.input_dim, self.hparams.model_dim, self.hparams.output_dim, self.hparams.dropout)
    def forward(self, x):
        x = self.batchlayer(x)
        z, h = self.autoencoder(x)
        return x, h
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class DeepGuardPredictor(DeepGuard):
    def _calculate_loss(self, batch, mode="train"):
        x, _ = batch
        x, h = self.forward(x)
        # loss = F.cross_entropy(x, h)
        loss = F.mse_loss(x, h)
        self.log("%s_loss" % mode, loss, on_epoch=True, enable_graph=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss
    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x, h = self.forward(x)
        # loss = F.cross_entropy(x, h)
        loss = F.mse_loss(x, h, reduce=False).mean(dim=1)
        return loss
