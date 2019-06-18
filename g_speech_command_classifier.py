import os
import os.path
import soundfile as sf
import librosa
import torch.utils.data as data
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch
from torch import optim
import numpy as np


# possible audio extensions
AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(file_name):
    """ checking if file is ends with one of the audio extensions """
    return any(file_name.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir_path):
    """ mapping each folder type of audio that we will classify in our task, in maximum can be 30 """
    classes = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir_path, class_to_idx=None):
    """ extracting data sets from folders which can be for train or for validation """
    spects = []
    full_dir = os.path.expanduser(dir_path)
    for target in sorted(os.listdir(full_dir)):
        d = os.path.join(full_dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, file_names in sorted(os.walk(d)):
            for file_name in sorted(file_names):
                if is_audio_file(file_name):
                    path = os.path.join(root, file_name)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len):
    """ extracting spectrograms from each audio file by using stft algorithm """
    y, sr = sf.read(path)
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandDataset(data.Dataset):
    """ creating pytorch dataset which can be of type training or validation, this dataset skeleton
        is helpful for later use when we will create data loaders from this data sets, just need to implement getitem
        and len functions """
    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)


def get_test_dataset(root):
    """ extracting test dataset from test folder and returning list of tuples of all audio files """
    spects = []
    full_dir = os.path.expanduser(root)
    for root, _, file_names in sorted(os.walk(full_dir)):
        for file_name in sorted(file_names):
            if is_audio_file(file_name):
                path = os.path.join(root, file_name)
                # each item/example is tuple
                item = (path, file_name)
                spects.append(item)
    return spects


def get_test_loader(test_dataset, window_size=.02, window_stride=.01, window_type='hamming',
                    normalize=True, max_len=101):
    """ extracting features from test dataset and creating test data loader"""
    test_loader = []
    for path, file_name in test_dataset:
        spect = spect_loader(path, window_size, window_stride, window_type, normalize, max_len).unsqueeze(0)
        test_loader.append(spect)
    return test_loader


class LeNet5(nn.Module):
    """ LeNet5 model which is old model we want to test """
    def __init__(self, num_classes=30, init_weights=True):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16280, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """ feed forward step of our model """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ initializing parameters of our model """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG(nn.Module):
    """ Generic VGG module """
    def __init__(self, features, num_classes=30, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7680, 512),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """ feed forward step of our model """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ initializing parameters of our model """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    """ creating cnn layers off vgg model depending on configuration which can be for (Vgg11 or Vgg13) """
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                       nn.BatchNorm2d(v),
                       nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


# configuration of vgg we can use, the number is for convolution layer and 'M' is for max pooling
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


def vgg11():
    """ creating vgg11 model """
    return VGG(make_layers(cfgs['vgg11']))


def vgg13():
    """ creating vgg13 model """
    return VGG(make_layers(cfgs['vgg13']))


# arguments
parser = argparse.ArgumentParser(description='LeNet5 for speech commands recognition task')
parser.add_argument('--train_path', default='data/train', help='path to the train data folder')
parser.add_argument('--valid_path', default='data/valid', help='path to the validation data folder')
parser.add_argument('--test_path', default='data/test', help='path to the test data folder')
parser.add_argument('--output_path', default='test_y', help='path to the output file')
parser.add_argument('--results_path', default='results.png', help='path to the results file')
parser.add_argument('--batch_size', type=int, default=16, help='training and valid and test batch size')
parser.add_argument('--epochs', type=int, default=32, help='number of epochs to train')
parser.add_argument('--patience', type=int, default=4, help='number of epochs of no loss improvement')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')


# parse args
args = parser.parse_args()


def train(train_loader, valid_loader, model, optimizer, criterion):
    """ training step where we training our model and evaluating him each epoch """
    best_valid_loss = np.inf
    patience = 0
    epoch = 0
    while epoch < args.epochs and patience < args.patience:
        model.train()
        correct = 0.
        acc_loss = 0.
        for data, labels in train_loader:
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, labels)
            acc_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            loss.backward()
            optimizer.step()
        # validation
        train_loss = acc_loss / len(train_loader.dataset)
        train_acc = 100 * correct / len(train_loader.dataset)
        print('Epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, train_acc))
        valid_loss, valid_acc = validate(valid_loader, model, criterion)
        print('Epoch: {}, Valid loss: {}, Valid accuracy: {}'.format(epoch, valid_loss, valid_acc))

        # checking for early stopping check
        if valid_loss > best_valid_loss:
            patience += 1
        else:
            patience = 0
            best_valid_loss = valid_loss
        epoch += 1


def validate(valid_loader, model, criterion):
    """ validation step which called each training epoch for checking our models results """
    model.eval()
    correct = 0.
    acc_loss = 0.
    for data, labels in valid_loader:
        data, labels = data.cuda(), labels.cuda()
        output = model.forward(data)
        acc_loss += criterion(output, labels).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

    valid_loss = acc_loss / len(valid_loader.dataset)
    valid_acc = 100 * correct / len(valid_loader.dataset)
    return valid_loss, valid_acc


def test(test_loader, model):
    """ testing step where we predicting via the model on test dataset """
    model.eval()
    preds = []
    for data in test_loader:
        data = data.cuda()
        output = model.forward(data)
        pred = output.data.max(1, keepdim=True)[1].item()
        preds.append(pred)
    return preds


def save(path_output, data, preds):
    """ saving predictions to output file """
    with open(path_output, 'w') as output:
        for i in range(len(data)):
            _, file_name = data[i]
            output.write('{}, {}\n'.format(file_name, preds[i]))


def main():
    """ ex4 exercise main function, which extracting data sets, building model's, training them and testing them """
    train_dataset = GCommandDataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = GCommandDataset(args.valid_path)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_dataset = get_test_dataset(args.test_path)
    test_loader = get_test_loader(test_dataset)

    torch.cuda.manual_seed(args.seed)
    # build model (LeNet5, VGG11 or VGG13)
    # model = torch.nn.DataParallel(LeNet5()).cuda()
    # model = torch.nn.DataParallel(vgg11()).cuda()
    model = torch.nn.DataParallel(vgg13()).cuda()
    # setting optimizer (SGD, SGD with momentum or ADAM)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # training
    train(train_loader, valid_loader, model, optimizer, criterion)
    # testing
    preds = test(test_loader, model)
    save(args.output_path, test_dataset, preds)


if __name__ == '__main__':
    main()
