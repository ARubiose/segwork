"""Simple nodes """
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from segwork.utils.mermaid.utils import lookahead

@dataclass
class MermaidNode(ABC):
    """Abstract base class for Memaid nodes
    """
    id:str
    text:str = ''

    @property
    @abstractmethod
    def representation(self):
        """Return string representation of node in Mermaid format"""

@dataclass
class MermaidNodeShape(MermaidNode):
    """Base Mermaid node class
    """
    shape:Tuple[str,str] = ('','') 

    def representation(self, end:str='\n'):
        content = self.text if self.text else self.id
        return f'{self.id}{self.shape[0]}{content}{self.shape[1]}{end}'

@dataclass
class MermaidNodeSubgraph(MermaidNode):
    """Base Mermaid subgraph class
    """

    direction:str = None
    children: list[MermaidNode] = field(default_factory=list)

    def representation(self, end:str='\n') -> str:
        chain = f'subgraph {self.id}{end}'
        chain.join( [f'\t{child.parse_node()}' for child in self.children] )
        return chain

@dataclass
class Mermaidlink(MermaidNode):
    """Base Mermaid link class. Single link"""

    shape:Tuple[str,str] = ('-','->') 
    source:list[Union[MermaidNode, str]] = field(default_factory=list)
    target:list[Union[MermaidNode, str]] = field(default_factory=list)

    def representation(self, end:str='\n') -> str:
        chain = self._concat_nodes(*self.source)
        chain += f'{self.shape[0]}{self.text}{self.shape[1]}'
        chain += self._concat_nodes(*self.target)
        return chain + end

    def _concat_nodes(self, *nodes:MermaidNode) -> str:
        chain = ''
        for s, has_more in lookahead(nodes):
            id = s if isinstance(s, str) else s.id
            chain += f'{id} '
            if has_more: chain += ' & '
        return chain
    
    def add_source(self, *nodes: MermaidNode) -> None:
        """Add source"""
        self.source.extend(list(nodes))

    def add_target(self, *nodes:MermaidNode) -> None:
        """Add target"""
        self.target.extend(list(nodes))
