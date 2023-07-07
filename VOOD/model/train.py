import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from config import settings, load_metadata, \
                    parse_datetime_strings
from VOOD.data.psd_loader import PSDDataset, min_max_scaler
from VOOD.model.dense_classification import MultiOutputSignalClassifier

T.manual_seed(1)

nperseg = 8192
model_name = 'model1'
training_params ='params1'
path_data = Path(settings.default.path['processed_data'])
database_path = path_data / f'PSD{nperseg}.db'
metadata_path = path_data / f'metadata_{nperseg}.json'
min_max_path = path_data / f'min_max_{nperseg}.json'

training_range = parse_datetime_strings(settings.split['train'])
validation_range = parse_datetime_strings(\
    settings.split['validation'])
testing_range = parse_datetime_strings(settings.split['test'])
metadata = load_metadata(metadata_path)
min_max = load_metadata(min_max_path)
min_max_scaler = min_max_scaler(**min_max)
model_params = settings.model_params[model_name]
training_params = settings.training[training_params]

general_ds_params = {'database_path': database_path,
                            'transform': min_max_scaler,
                            'drop': ['ACC1_X', 'ACC1_Y'],
                            'preload': True}

ds_tr = PSDDataset(ts_start =training_range['start'],
                ts_end =training_range['end'],
                **general_ds_params)
ds_val= PSDDataset(ts_start =validation_range['start'],
                ts_end =validation_range['end'],
                **general_ds_params)

dl_tr = DataLoader(ds_tr, 
                batch_size=training_params['batch_size'],
                shuffle=True, num_workers=4, pin_memory=True)
dl_val = DataLoader(ds_val,
                batch_size=training_params['batch_size'],
                shuffle=False, num_workers=4, pin_memory=True)
input_dim =len(list(ds_tr.data.values())[0][0])
print(input_dim)
net = MultiOutputSignalClassifier(input_dim=input_dim,
                                  **model_params)
optimizer = T.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                         weight_decay=training_params['weight_decay'],
                         momentum=training_params['momentum'],)
scheduler = T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            T_0=training_params['epochs'],
            T_mult= training_params['T_mult'],
            eta_min=training_params['learning_rate']/16)




def train_epochs(net, optimizer, train_loader, scheduler):
    net.train()
    loss_avg = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        psds = data[0]

        y_position = nn.functional.one_hot(data[2], 4).float()
        y_direction = nn.functional.one_hot(data[1], 3).float()

        output = net(psds)
        position_output = output[0]
        direction_output = output[1]
        loss = nn.CrossEntropyLoss()(position_output, y_position) + \
            nn.CrossEntropyLoss()(direction_output, y_direction)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

def validate(net, valid_loader):
    net.eval()
    loss_avg = 0.0
    correct_positions = 0
    correct_directions = 0
    total = 0
    with T.no_grad():
        for data in valid_loader:
            psds = data[0]
            y_direction = nn.functional.one_hot(data[1], 3).float()
            y_position = nn.functional.one_hot(data[2], 4).float()

            output = net(psds)
            position_output = output[0]
            direction_output = output[1]
            loss = nn.CrossEntropyLoss()(position_output, y_position) + \
                nn.CrossEntropyLoss()(direction_output, y_direction)
            loss_avg += loss.item()

            _, predicted_positions = T.max(position_output.data, 1)
            _, predicted_directions = T.max(direction_output.data, 1)



            total += y_position.size(0)
            correct_positions += (predicted_positions == data[2]).sum().item()
            correct_directions += (predicted_directions == data[1]).sum().item()
        avg_loss = loss_avg / len(valid_loader)
        position_accuracy = correct_positions / total
        direction_accuracy = correct_directions / total
    return avg_loss, position_accuracy, direction_accuracy

def train(net, optimizer, train_loader, valid_loader, scheduler):
    for epoch in range(training_params['epochs']):
        train_epochs(net, optimizer, train_loader, scheduler)
        avg_loss, position_accuracy, direction_accuracy = validate(net, valid_loader)
        print(f'Epoch {epoch}: loss {avg_loss}, position accuracy {position_accuracy}, direction accuracy {direction_accuracy}')

if __name__ == '__main__':
    train(net, optimizer, dl_tr, dl_val, scheduler)
