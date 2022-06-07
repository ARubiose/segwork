import torch.nn as nn
from .mermaid import TorchFXParser

def show_diagram( module: nn.Module):
    parser = TorchFXParser( name= module.__class__.__name__, module = module)
    parser.display_graph()