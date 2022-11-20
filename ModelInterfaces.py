
from pathlib import Path
from typing import Callable, Generator
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

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epoch_range: range = range(50), use_tqdm: str = "notebook") \
        -> Generator[nn.Module, None, None]:
        """
            Train the model using `train_dataloader` to load training set
            and `val_dataloader` to load testing/validation set.\n
            `epoch_range` defines the starting and ending epoch number e.g. `range(35,50)`. Default `range(50)`\n
            After each iteration, it will return a model snapshot; use a for-loop to wait for an iteration to complete.
            ```py
            for model_snapshot in my_IModel_object.train(...):
                torch.save(model_snapshot, '/path/model_epoch')
            ```
            `use_tqdm` allows you to specify whether you want to have HTML-styled progress bar ("notebook")
            or in-console progress bar ("console"), or no progress bar at all (None). Default "notebook".
        """
        raise NotImplementedError("Training Interface Method Not Implemented")
    
    def infer(self, dataloader: DataLoader, confidence_threshold: float = 0.5) -> str:
        """
            Runs inferences on the provided i3D-extracted videos `dataloader`, generates a JSON report
            in a similar structure like the original TSU JSON files.\n
            Frames with a classification probability below `confidence_threshold` will be excluded. Default 0.5.\n
            Returns the path at which the report was saved.
        """
        raise NotImplementedError("Inference Interface Method Not Implemented")

    def evaluate(self, dataloader: DataLoader):
        """
            calculates mAP score and average class predictions of the model
            given the data subset in `dataloader`
        """
        raise NotImplementedError("Evaluation Interface Method Not Implemented")
