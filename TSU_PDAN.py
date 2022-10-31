from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from ModelInterfaces import IModel
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
        # TODO:rewrite in a way that can call progress
        HOI.train.run(
            [(self.model, 0, {'train': train_dataloader, 'val': val_dataloader}, optimizer, lr_sched, self.comp_info)],
            criterion,
            num_epochs=epochs,
            modelkind="PDAN",
            lr=self.lr
        )

    def infer(self, dataloader: DataLoader):
        pass
    def evaluate(self, dataloader: DataLoader):
        pass
