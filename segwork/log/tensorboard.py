import typing
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """Class to save training logs on TensorBoard"""
    
    def __init__(self):
        self.writer = SummaryWriter()

    def add_scalar(
            self,
            name:typing.Union[typing.List[str], str], 
            value: typing.Union[typing.List[typing.Any], typing.Any],
            n_iter:int):
        if not isinstance(name, typing.List):
            pass
        self.writer.add_scalar()