from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from ModelInterfaces import IModel

# TODO: fix TSU's code being hard-coded to check command line arguments
import sys
sys.argv = sys.argv[0:1] + [
    "-batch_size", "1",
    "-APtype", "map",
    "-num_classes", "51",
    "-model", "PDAN"
]

import HOI.train
import HOI.models

class HOI_PDAN(IModel):
    def __init__(self, mode: str = "rgb", num_channel: int = 512, num_classes: int = 51) -> None:
        super().__init__()
        if mode == "skeleton":
            input_channel = 256
        elif mode == "rgb":
            input_channel = 1024
        else:
            raise ValueError("mode may only be skeleton or rgb")

        model = HOI.models.PDAN(num_stages=1, num_layers=5, num_f_maps=num_channel, dim=input_channel, num_classes=num_classes)
        model = nn.DataParallel(model)
        model = model.cuda()
        self.model = model

        self.output_directory = "./PDAN/"

    def PDAN_training_parameters(self, lr: float = 0.0002, comp_info: str = "TSU_CS_RGB_PDAN"):
        """Set training parameters specific to Toyota HOI"""
        self.lr = lr
        self.comp_info = comp_info

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 50):
        criterion = nn.NLLLoss(reduce=False)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        model_algo = "PDAN"
        best_map = 0.0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            probs = []
            train_map, train_loss = HOI.train.train_step(self.model, 0, optimizer, train_dataloader, epoch)
            prob_val, val_loss, val_map = HOI.train.val_step(self.model, 0, val_dataloader, epoch)
            probs.append(prob_val)
            lr_sched.step(val_loss)

            self.progress_callback.__func__(epoch, epochs)

            if best_map < val_map:
                best_map = val_map
                torch.save(self.model.state_dict(),'./'+model_algo+'/weight_epoch_'+str(self.lr)+'_'+str(epoch))
                torch.save(self.model,'./'+model_algo+'/model_epoch_'+str(self.lr)+'_'+str(epoch))
                print('save here:','./'+model_algo+'/weight_epoch_'+str(self.lr)+'_'+str(epoch))

    def infer(self, dataloader: DataLoader):
        pass
    def evaluate(self, dataloader: DataLoader):
        pass
