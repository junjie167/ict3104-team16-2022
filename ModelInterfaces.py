
from typing import Callable
from torch.utils.data import DataLoader
from inspect import signature

class IModel:

    CallbackType = Callable[[int, int], None]
    __progress_callback = lambda current_index, total_indexes: None

    @property
    def progress_callback(self) -> CallbackType:
        """
            Set this property to a callable (function or lambda)
            with signature `callback(current_index: int, total_indexes: int)`.\n
            E.g. `TSU_PDAN.progress_callback = lambda current, total: print(f'Training {(current/total) * 100}%')`
        """
        return self.__progress_callback;

    @progress_callback.setter
    def __set_progress_callback(self, callback: CallbackType):
        callback_params = signature(callback).parameters
        if(len(callback_params) != 2):
            raise TypeError("callback must have 2 parameters")

        self.__progress_callback = callback;

    def train(self, dataloder: DataLoader):
        pass
    
    def infer(self, dataloader: DataLoader):
        pass

    def evaluate(self, dataloader: DataLoader):
        pass
