from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from VOOD.data.psd_loader import PSDDataset, min_max_scaler
from config import settings, load_metadata, parse_datetime_strings
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nperseg',type=int, default=8192)
parser.add_argument('--model_name', type=str, default='model1') 
parser.add_argument('--training_params', type=str, default='params1')
args = parser.parse_args()


path_data = Path(settings.default.path['processed_data'])
database_path = path_data / f'PSD{args.nperseg}.db'
min_max_path = path_data / f'min_max_{args.nperseg}.json'
metadata_path = path_data / f'metadata_{args.nperseg}.json'
min_max = load_metadata(min_max_path)
training_params = settings.training[args.training_params]


min_max_scaler_transformer = min_max_scaler(**min_max)
settings_dl = {'database_path': database_path,
                            'transform': min_max_scaler_transformer,
                            'drop': ['ACC1_X', 'ACC1_Y'],
                            'preload': True}
def accuracy(y_pred_pos,y_pred_dir,
                y_true_pos,y_true_dir):
    y_pred_pos = torch.argmax(y_pred_pos, dim=1)
    y_pred_dir = torch.argmax(y_pred_dir, dim=1)
    position_acc = torch.sum(y_pred_pos == y_true_pos).item() / len(y_pred_pos)
    direction_acc = torch.sum(y_pred_dir == y_true_dir).item() / len(y_pred_dir)
    return position_acc, direction_acc

def confusion_matrix(y_pred_pos,y_pred_dir,
                y_true_pos,
                y_true_dir):
    y_pred_pos = torch.argmax(y_pred_pos, dim=1)
    y_pred_dir = torch.argmax(y_pred_dir, dim=1)

    position_matrix = torch.zeros(4,4)
    direction_matrix = torch.zeros(3,3)
    for i in range(4):
        for j in range(4):
            position_matrix[i,j] = torch.sum((y_pred_pos == i) & (y_true_pos == j))
            direction_matrix[i,j] = torch.sum((y_pred_dir == i) & (y_true_dir == j))
    return position_matrix, direction_matrix
class MultiOutputSignalClassifier(pl .LightningModule):
    def __init__(self, 
                 num_positions,
                 num_directions, 
                 input_dim, 
                 dense_layers,
                 dropout_rate=0.2, 
                 batch_norm=True, 
                 use_bias=True,
                 activation='relu', 
                 l1_reg=0.01, 
                 temperature=1.0):
        
        super(MultiOutputSignalClassifier, self).__init__()

        # Initializing class variables
        self.num_positions = num_positions
        self.num_directions = num_directions
        self.input_dim = input_dim
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l1_reg = l1_reg
        self.temperature = temperature
        self.use_bias = use_bias
        self.save_hyperparameters()
        # Defining activation functions
        activation_functions = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        self.activation_fn = activation_functions.get(activation.lower(), nn.ReLU())

        # Building the hidden layers
        self.hidden_layers = nn.ModuleList()
        for units in self.dense_layers:
            self.hidden_layers.append(nn.Linear(self.input_dim, units, bias=self.use_bias))
            self.input_dim = units

        # Defining the final output layers
        self.position_layer = nn.Linear(self.input_dim, self.num_positions)
        self.direction_layer = nn.Linear(self.input_dim, self.num_directions)

        # Additional layers for batch normalization and dropout
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(units) for units in self.dense_layers])
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation_fn(x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = self.dropout_layer(x)
        return x
    
    def _comon_step(self,batch):
        x, y_direction, y_position = batch
        y_direction = F.one_hot(y_direction, self.num_directions).float()
        y_position = F.one_hot(y_position, self.num_positions).float()
        output =self(x)
        position_output = self.position_layer(output)
        direction_output = self.direction_layer(output)

        loss = nn.CrossEntropyLoss()(position_output, torch.argmax(y_position, dim=1)) + \
            nn.CrossEntropyLoss()(direction_output, torch.argmax(y_direction, dim=1))
        return loss , position_output, direction_output
    

    
    
    def training_step(self, batch, batch_idx):
        loss,_,_ = self._comon_step(batch)
        self.log('train_loss', loss)
        return loss
     
    
    def validation_step(self, batch, batch_idx):
        x, y_direction, y_position = batch
        loss, position_output, direction_output = self._comon_step(batch)
        self.log('val_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        #compute accuracy
        position_acc, direction_acc = accuracy(position_output,
                                            direction_output, 
                                            y_position, 
                                            y_direction)
        self.log('val_position_acc', position_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_direction_acc', direction_acc, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_direction, y_position = batch
        loss, position_output, direction_output = self._comon_step(batch)
        self.log('test_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        #compute accuracy
        position_acc, direction_acc = accuracy(position_output,
                                            direction_output,
                                            y_position,
                                            y_direction)
        self.log('test_position_acc', position_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_direction_acc', direction_acc, on_epoch=True, prog_bar=True, logger=True)    
        return {'loss':loss,
                'position_output':position_output,
                'direction_output':direction_output,
                'y_position':y_position,
                'y_direction':y_direction}

    def on_test_epoch_end(self, outputs):
        concat_position_output = torch.cat([x['position_output'] for x in outputs])
        concat_direction_output = torch.cat([x['direction_output'] for x in outputs])
        concat_y_position = torch.cat([x['y_position'] for x in outputs])
        concat_y_direction = torch.cat([x['y_direction'] for x in outputs])
        cf_position , cf_direction = confusion_matrix(concat_position_output,
                                                        concat_direction_output,
                                                        concat_y_position,
                                                        concat_y_direction)
        self.logger.experiment.add_figure('confusion_matrix_position', cf_position)
        self.logger.experiment.add_figure('confusion_matrix_direction', cf_direction)
        return {'confusion_matrix_position':cf_position,
                'confusion_matrix_direction':cf_direction}

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(),
                                lr=training_params['learning_rate'])
    
    def train_dataloader(self):

        training_range = parse_datetime_strings(settings.split['train'])
        
        ds_tr = PSDDataset(ts_start= training_range['start'],
                           ts_end = training_range['end'],
                            **settings_dl)
        print(f'expected input dimension : {len(list(ds_tr.data.values())[0][0])}')    
        dl_tr = DataLoader(ds_tr, 
                           batch_size=128, 
                           shuffle=True, num_workers=8)
        return dl_tr
    
    def val_dataloader(self):
        validation_range = parse_datetime_strings(settings.split['validation'])
        ds_val = PSDDataset(ts_start= validation_range['start'],
                            ts_end = validation_range['end'],
                             **settings_dl)
        dl_val = DataLoader(ds_val,
                            batch_size=128,
                            shuffle=False, num_workers=4)
        return dl_val
    def test_dataloader(self):
        test_range = parse_datetime_strings(settings.split['test'])
        ds_test = PSDDataset(ts_start= test_range['start'],
                             ts_end = test_range['end'],
                             **settings_dl)
        dl_test = DataLoader(ds_test,
                             batch_size=128,
                             shuffle=False, num_workers=4)
        return dl_test
    
def get_callbacks():
    early_stop_callback = pl.callbacks\
                            .EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=10,
                                        verbose=True,
                                        mode='min')
    model_summary = pl.callbacks.ModelSummary(max_depth=-1)
   
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = Path(training_params['save'])/args.model_name,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    return [early_stop_callback, model_summary,checkpoint_callback,
            lr_monitor]
    
        
if __name__ == '__main__':
    model_params = settings.model_params[args.model_name]
    logger = pl.loggers.TensorBoardLogger(save_dir='logs/lightning_logs', name=args.model_name)
    trainer = Trainer(max_epochs=training_params['epochs']-95,
                      logger=logger,
                      callbacks=get_callbacks())
    model = MultiOutputSignalClassifier(input_dim=1632,**model_params)
    trainer.fit(model)

    # %tensorboard --logdir logs/lightning_logs

