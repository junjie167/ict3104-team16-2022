
from pathlib import Path
from typing import Callable
from torch.utils.data import DataLoader
import torch.nn as nn
from inspect import signature

class IModel:

    CallbackType = Callable[[int, int], None]
    __progress_callback = lambda current_index, total_indexes: None
    __torch_model = None
    __directory = "./unknown_model"

    @property
    def progress_callback(self) -> CallbackType:
        """
            Set this property to a callable (function or lambda)
            with signature `callback(current_index: int, total_indexes: int)`.\n
            E.g. `TSU_PDAN.progress_callback = lambda current, total: print(f'Training {(current/total) * 100}%')`
        """
        return self.__progress_callback;

    @progress_callback.setter
    def progress_callback(self, callback: CallbackType):
        callback_params = signature(callback).parameters
        if len(callback_params) != 2:
            raise TypeError("callback must have 2 parameters")

        self.__progress_callback = callback;

    @property
    def model(self) -> nn.Module:
        """
            Torch.nn model.\n
            If you wish to load existing model, just do:\n
            ```py
            my_IModel_object.model.load_state_dict(
                torch.load(path_to_model)
            )
            ```
        """
        if self.__torch_model is None:
            raise NotImplementedError("no torch model defined")
        return self.__torch_model

    @model.setter
    def model(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError(f"expected torch model, got {type(model)}")
        self.__torch_model = model;

    @property
    def output_directory(self) -> str:
        Path(self.__directory).mkdir(parents=True, exist_ok=True)
        return self.__directory
    
    @output_directory.setter
    def output_directory(self, dir: str):
        self.__directory = dir

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        raise NotImplementedError("Training Interface Method Not Implemented")
    
    def infer(self, dataloader: DataLoader):
        raise NotImplementedError("Inference Interface Method Not Implemented")

    def evaluate(self, dataloader: DataLoader):
        raise NotImplementedError("Evaluation Interface Method Not Implemented")
