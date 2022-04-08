from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
from functools import partial
import logging
import os

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from IPython.display import Image, display

from segwork.utils.mermaid.style import MermaidStyle, TorchFxStyle, Styles
from segwork.utils.mermaid.node import MermaidNode
from segwork.utils.mermaid.token import MermaidToken, MermaidTokenFunction, MermaidTokenPlaceholder

logger = logging.getLogger(__name__)
       
@dataclass
class MermaidGraph:

    name:str
    type:str = 'flowchart'
    orientation:str = 'TB'
    nodes:list[MermaidNode] = field(default_factory=list)

    def __post__init__(self):
        if self.orientation not in Styles.ORIENTATION_TYPES:
            self.orientation = 'TB'
            raise ValueError(f'Invalid orientation {self.orientation}. \
                                Set default: TB \
                                Valid orientations: {Styles.ORIENTATION_TYPES}')

    def _save(self, path:str=None):
        """Generte text document of diagram"""
        path = path or os.path.join(os.getcwd(), 'mermaid.md')
        try:
            with open(path, 'x') as f:
                f.write(f'{self.type} {self.orientation}\n') # Type and orientation
                f.write(self.create_title()) # Title
                for node in self.nodes: # Nodes and links
                    f.write(f'\t{node.representation()}')
        except FileExistsError as e:
            logger.warn('File {path} already exist. Delete the file or change name before saving.')
        return True

    def _display(self, path:str=None):
        """Display graph
        
        #FIXME Create raw string and display"""
        path = path or os.path.join(os.getcwd(), 'mermaid.md')
        try:
            with open(path, 'r') as f:
                graph = f.read()
                graphbytes = graph.encode("ascii")
                base64_bytes = base64.b64encode(graphbytes)
                base64_string = base64_bytes.decode("ascii")
                display(Image(url="https://mermaid.ink/img/" + base64_string))
                
        except FileNotFoundError as e:
            logger.info(f'File {path} not found. Saving file...')
            self._save(path)
            self._display(path)


    def add_child(self, *nodes:MermaidNode) -> None:
        for node in nodes:
            assert isinstance(node, MermaidNode), f'Element must be a MermaidNode. Got {node.__class__.__name__,}'
            self.nodes.append(node)

    def create_title(self) -> str:
        """Return styled title in str format"""
        return f'\ttitle[<u>{self.name}</u>]\nstyle title fill:#FFF,stroke:#FFF\n'
        
class MermaidParser(ABC):
    """Base abstract class"""

    def __init__(self, name:str, style_map:MermaidStyle) -> None:
        self.style_map = style_map 
        self._tokens:list[MermaidToken]= list()
        self.diagram = MermaidGraph(name=name, type=style_map.diagram , orientation=style_map.orientation)
        
    @property
    def tokens(self):
        return self._tokens
        
    def parse_tokens(self):
        """Populate diagram with nodes"""
        assert not self.diagram.nodes, "Token list is not empty. Reset parser before."
        for token in self._tokens:
            nodes = token.parse()
            self.diagram.add_child(*nodes)
    
    def reset(self) -> None:
        """Create new graph and empty token list"""
        self._tokens = list()
        self.diagram = MermaidGraph()

    @abstractmethod
    def tokenize(self, **kwargs):
        """Populate tokens attribute"""
        pass

    def display_graph(self, path:str=None):
        self.diagram._display(path)

    def save_graph(self, path:str=None):
        self.diagram._save(path)


class TorchFXParser(MermaidParser):

    TOKEN_MAP = {
        'placeholder':MermaidTokenPlaceholder,
        'call_module':partial(MermaidTokenFunction, is_module=True),
        'call_function':MermaidTokenFunction,
        'output': MermaidTokenFunction
    }

    def __init__(self, name:str, style_map:MermaidStyle = None, module:nn.Module = None, **kwargs)-> None:
        if not style_map:
            style_map = TorchFxStyle()
        super().__init__(name, style_map)
        if module: self.tokenize(module)

    def tokenize(self, module:nn.Module)-> None:
        self._module = module
        torch_nodes = symbolic_trace(module).graph.nodes
        self._tokens = [self.create_token(n) for n in torch_nodes]

    def create_token(self, data:torch.fx.Node, **kwargs) -> MermaidToken:
        """Create token from torch fx node and style mapping"""
        style = getattr(self.style_map, data.op, {})
        token_class = self.TOKEN_MAP.get(data.op, MermaidTokenPlaceholder)
        return token_class(data, style, self._module)

def display_model(model:nn.Module):
    """Display model"""
if __name__ == '__main__':
    pass