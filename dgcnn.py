import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import argparse
from torcheeg import transforms
from torcheeg.datasets import SEEDIVDataset, SEEDIVFeatureDataset, SEEDDataset, SEEDFeatureDataset
from torcheeg.datasets.constants.emotion_recognition.seed_iv import SEED_IV_STANDARD_ADJACENCY_MATRIX
from torcheeg.datasets.constants.emotion_recognition.seed import SEED_ADJACENCY_MATRIX
from torcheeg.models.pyg import RGNN
from torcheeg.models import DGCNN
from torcheeg.transforms.pyg import ToG
from torcheeg.trainers import ClassificationTrainer
from torch_geometric.loader import DataLoader
from torcheeg.model_selection import KFoldGroupbyTrial
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Tuple
from rich.progress import track
from rich import print
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
parser = argparse.ArgumentParser()
parser.add_argument("epochs",type=int,help="Number of epochs in training")
parser.add_argument("train_path",type=str,help="Path to training")
args = parser.parse_args()
epochs = args.epochs
train_type = args.train_path
file_path = os.path.join("./",train_type,"model.pth")
print(f'DEVICE USED: {device}')
print(f'DEVICE USED: {device}')


class MyClassificationTrainer(ClassificationTrainer):
    def __init__(self, model: nn.Module, trainer_k:str=None, num_classes=None, lr: float = 0.0, weight_decay: float = 0,optimizer = None, **kwargs):
        super().__init__(model, num_classes, lr, weight_decay, **kwargs)
        self.writer = SummaryWriter(f"./{train_type}/train_{trainer_k}/loss")
        self.steps_file_name =f"./{train_type}/train_{trainer_k}/steps"
        self.train_counter = 0
        self.test_counter = 0
        self.trainer_k = trainer_k
        self.optimizer = optimizer
        self.epoch = 0
        self.last_train_loss = 0
        self.last_train_accuracy = 0
    
    def on_training_step(self, train_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        super().on_training_step(train_batch, batch_id, num_batches, **kwargs)
        if self.train_loss.mean_value.item() != 0:
            self.last_train_loss = self.train_loss.compute()
            self.last_train_accuracy = self.train_accuracy.compute()

    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        super().after_validation_epoch(epoch_id, num_epochs, **kwargs)
        torch.save(model.state_dict(),f"./{train_type}/train_{self.trainer_k}/model.pth")
        self.writer.add_scalars('loss', {
            'train': self.last_train_loss,
            'validation': self.val_loss.compute()
        }, self.train_counter)
        self.writer.add_scalars('accuracy', {
            'train': self.last_train_accuracy*100,
            'validation': self.val_accuracy.compute()*100
        }, self.train_counter)
        self.train_counter += 1
        self.epoch += 1
    
    def after_test_epoch(self, **kwargs):
        super().after_test_epoch(**kwargs)
        self.writer.add_scalar("loss/test", self.test_loss.compute(), self.test_counter)
        self.writer.add_scalar("accuracy/test", self.test_accuracy.compute()*100, self.test_counter)
        self.test_counter += 1




if __name__ == '__main__':
    
    dataset = SEEDIVDataset(io_path=f'./dataset/seed_iv_RAW_DATA', root_path='./dataset/raw_data',
                            offline_transform=transforms.Concatenate([
                            
                            transforms.BandDifferentialEntropy(band_dict={
                                "beta": [14,31]
                            }),
                            
                            transforms.BandPowerSpectralDensity(band_dict={
                                "beta": [14,31]
                            })
                            
                            ]),

                            online_transform=transforms.Compose([
                                #transforms.MeanStdNormalize(),
                                transforms.ToTensor(),
                                
                            ]),
                            label_transform=transforms.Select('emotion'),
                            chunk_size=800, num_worker=8)
    
    """
    # Carica il dataset di feature
    dataset = SEEDIVFeatureDataset(io_path=f'./dataset/seed_iv_2', root_path='./dataset/eeg_feature_smooth',
                                   feature=['de_LDS'],
                                   online_transform=transforms.Compose([
                                       transforms.MeanStdNormalize(),transforms.ToTensor()
                                   ]),
                                   label_transform=transforms.Select('emotion'))
    
    """
    """
    dataset = SEEDDataset(io_path=f'./dataset/seed',
                          root_path='./dataset/Preprocessed_EEG',
                      offline_transform=transforms.BandDifferentialEntropy(band_dict={
                          "delta": [1, 4],
                          "theta": [4, 8],
                          "alpha": [8, 14],
                          "beta": [14, 31],
                          "gamma": [31, 49]
                      }),
                      online_transform=transforms.Compose([
                          transforms.ToTensor()
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select('emotion')
                      ]))
    """
                            

    k_fold = KFoldGroupbyTrial(n_splits=5,shuffle=True,random_state=10,split_path='./dataset/splits_2')
    # Addestramento della rete neurale
    for i, (train_dataset, val_dataset) in track(enumerate(k_fold.split(dataset)), "[bold green]Training: ", total=5):
        model = DGCNN(num_electrodes=62, in_channels=2, num_layers=2, hid_channels=32, num_classes=4).to(device)
        trainer = MyClassificationTrainer(model=model, trainer_k = i,optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, weight_decay=0.0001))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        trainer.fit(train_loader, val_loader, num_epochs=epochs)    
        trainer.test(val_loader)


    print('[bold green]Addestramento completato!')
