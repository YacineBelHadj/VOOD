from VOOD.model.dense_classification import MultiOutputSignalClassifier
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import argparse
import torch

# parse settings from command line
parser = argparse.ArgumentParser(description='Dense Classification Training with Energy Loss',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--energy_loss', '-el', type=float, default=0.01, help='Energy loss weight.')

args = parser.parse_args()

args.save = args.save + 'dense_classification_energy_loss'
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)

net= MultiOutputSignalClassifier(num_positions=4, 
                                 num_directions=3, 
                                 input_dim=3, 
                                 dense_layers=[1024,248,128,128,64, 64, 32,20],
                                dropout_rate=0, batch_norm=True,
                                activation='relu', l1_reg=0.01, temperature=1.0)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + F.cos(step / total_steps * F.pi))
optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lambda step: cosine_annealing(
                                                    step,
                                                    args.epochs * len(train_loader_in),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.learning_rate))

def train():
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_loader_in):
        y_position = target[0]
        y_direction = target[1]

        emb = net(data)
        loss = F.cross_entropy(emb, y_position) + F.cross_entropy(emb, y_direction)
        loss += state['energy_loss'] * (emb.mean(1) - F.logsumexp(emb, dim=1).mean())
        loss.backward()
        optimizer.step()
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state['train_loss'] = loss_avg

def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            output = net(data)
            y_position = target[0]
            y_direction = target[1]
            loss = F.cross_entropy(output, y_position) + F.cross_entropy(output, y_direction)
            pred_dir = output[0].max(1)[1]
            pred_pos = output[1].max(1)[1]
            correct += pred_dir.eq(y_direction.view_as(pred_dir)).sum().item() + pred_pos.eq(y_position.view_as(pred_pos)).sum().item()



