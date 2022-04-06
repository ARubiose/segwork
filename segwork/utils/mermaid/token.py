from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from segwork.utils.mermaid.node import MermaidNode, MermaidNodeShape, Mermaidlink

@dataclass
class MermaidToken(ABC):
    """Abstract base class for Mermaid Tokens. Relation between Mermaid nodes and tokens"""

    data:Any
    style:dict

    @abstractmethod
    def parse(self, **kwargs):
        """Parse token to Mermaid node"""
        pass

@dataclass
class MermaidTokenPlaceholder(MermaidToken):
    """Parse placeholder operation"""

    module:nn.Module
    name:str = 'placeholder'

    def parse(self, **kwargs) -> MermaidNodeShape:
        params = self.get_params()
        return [MermaidNodeShape(**params)]

    def get_params(self) -> Dict:
        id = self.data.name
        return {
            'id':id,
            **self.style
        }

@dataclass
class MermaidTokenFunction(MermaidToken):
    """Parse placeholder operation"""

    module:nn.Module
    is_module:bool = False
    name:str = 'call_module'

    def parse(self, **kwargs) -> list[MermaidNode]:
        params = self.get_params()
        module_node = MermaidNodeShape(**params['node'])

        link_node = Mermaidlink(**params['link'])
        source_nodes = [t.name for t in self.data.all_input_nodes]
        target_nodes = self.data.name
        link_node.add_source(*source_nodes)
        link_node.add_target(target_nodes)

        return [module_node, link_node]

    def get_params(self) -> Dict:
        # Node paramas
        text = self._get_module_info(self.data.target) if self.is_module else None
        id = self.data.name
        node_params = {
            'id':id,
            'text':text,
            **self.style['node']
        }
        #Link params
        link_params={
            'id': f'{id}-Link',
            **self.style['link']
        }
        return{
            'node':node_params,
            'link':link_params
        }

    def _get_module_info(self, name) -> Dict:
        #TODO Complete information
        module = self.module.get_submodule(name)
        info = module.__str__()
        info = info.replace('(', '[')
        info = info.replace(')', ']')
        return '\"' + info + '\"'

