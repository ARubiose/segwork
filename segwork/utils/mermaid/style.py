from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

class Styles():
    """Style Enum class"""
    DIAGRAM_TYPES = ['flowchart']
    ORIENTATION_TYPES = ['TB', 'TD', 'BT', 'RL', 'LR']

    #Shapes
    ROUND_EDGES :Tuple= ('(',')')
    RHOMBUS :Tuple= ('{', '}')
    ASYMMETRIC_SHAPE :Tuple= ('>', ']')

    #Link
    ARROW_LINK :Tuple= ('-', '->')
    DOTTED_LINK :Tuple=('-.', '.->')

@dataclass
class MermaidStyle:
    """Base class for Mermaid Style"""
    diagram:str 
    orientation:str

@dataclass
class TorchFxStyle(MermaidStyle):   
    """Style class for TorchFX wrapper"""
    
    diagram:str = 'flowchart'
    orientation:str = field(default='TB')

    placeholder = {'shape':Styles.ROUND_EDGES}

    call_module = {'node':{'shape':Styles.ROUND_EDGES}, 
                   'link':{'shape':Styles.ARROW_LINK}}

    call_function = {'node':{'shape':Styles.RHOMBUS}, 
                     'link':{'shape':Styles.DOTTED_LINK}}
    
    output = {'node':{'shape':Styles.ASYMMETRIC_SHAPE},
              'link':{'shape':Styles.ARROW_LINK}}

    # get_attr = {'type':'', 'style':{}}
    # call_method = {'type':'', 'style':{}}





    
