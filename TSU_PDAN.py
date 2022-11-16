from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
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
        
        self.__num_classes__ = num_classes

        model = HOI.models.PDAN(num_stages=1, num_layers=5, num_f_maps=num_channel, dim=input_channel, num_classes=num_classes)
        model = nn.DataParallel(model)
        model = model.cuda()
        self.model = model

        self.output_directory = "./PDAN/"
        self.epoch = 0

    def PDAN_training_parameters(self, lr: float = 0.0002, comp_info: str = "TSU_CS_RGB_PDAN"):
        """Set training parameters specific to Toyota HOI"""
        self.lr = lr
        self.comp_info = comp_info

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epoch_range: range = range(50), use_tqdm: str = "notebook"):
        criterion = nn.NLLLoss(reduce=False)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        model_algo = "PDAN"
        best_map = 0.0
        dataloaders = {
            "train": train_dataloader,
            "test": val_dataloader
        }

        from tqdm import tqdm
        from tqdm.notebook import tqdm_notebook
        for epoch in epoch_range:
            # print('Epoch {}/{}'.format(epoch, epochs - 1))
            # print('-' * 10)
            # due to a bug in TQDM, the progress bars must be re-constructed to reset them to zero
            if use_tqdm == "notebook":
                dataloaders["train"] = tqdm_notebook(train_dataloader, unit='batch', desc='training', leave=False)
                dataloaders["test"] = tqdm_notebook(val_dataloader, unit='batch', desc='validating', leave=False)
            elif use_tqdm == "console":
                dataloaders["train"] = tqdm(train_dataloader, unit='batch', desc='training', leave=False)
                dataloaders["test"] = tqdm(val_dataloader, unit='batch', desc='validating', leave=False)

            probs = []
            train_map, train_loss = HOI.train.train_step(self.model, 0, optimizer, dataloaders["train"], epoch)
            prob_val, val_loss, val_map = HOI.train.val_step(self.model, 0, dataloaders["test"], epoch)
            probs.append(prob_val)
            lr_sched.step(val_loss)

            # self.progress_callback.__func__(epoch, epochs)

            if best_map < val_map:
                best_map = val_map
                # torch.save(self.model.state_dict(),'./'+model_algo+'/weight_epoch_'+str(self.lr)+'_'+str(epoch))
                # torch.save(self.model,'./'+model_algo+'/model_epoch_'+str(self.lr)+'_'+str(epoch))
                # print('save here:','./'+model_algo+'/weight_epoch_'+str(self.lr)+'_'+str(epoch))
                self.epoch = epoch
                yield self.model

    def infer(self, dataloader: DataLoader):
        results = HOI.train.eval_model(self.model, dataloader)
        print("eval done, generating report")
        results = {
            video_name: {
                "actions": self.__compact_result__(values[1])
            } \
                for video_name, values in results.items()
        }
        # TODO: Change path
        with open("logga.json", mode="w") as logfile:
            json.dump(results, fp=logfile)

    def evaluate(self, dataloader: DataLoader):
        full_probs, epoch_loss, mAP_acc = HOI.train.val_step(self.model, 0, dataloader, self.epoch)
        return mAP_acc
    
    def __compact_result__(self, per_class_probs_per_frame: np.ndarray) -> list[dict]:
        # TODO: rewrite to show top 3
        # TODO: exclude low-confidence results?
        actions = list()
        longest = 0
        top_class = np.argmax(per_class_probs_per_frame, axis=1)
        for class_id in range(self.__num_classes__):
            class_idx = np.where(top_class == class_id)[0]
            if(len(class_idx) == 0):
                continue
            ranges = np.split(class_idx, np.where(np.diff(class_idx) != 1)[0]+1)
            for period in ranges:
                if len(period) > longest:
                    longest = len(period)
                actions.append({
                    "class": class_id,
                    "start": int(period[0]),
                    "end": int(period[-1]),
                    "confidence": float(np.mean(per_class_probs_per_frame[period[0]:period[-1]+1, class_id]))
                })
        actions.sort(key=lambda a: a["start"])
        # print("longest john", longest)
        return actions
