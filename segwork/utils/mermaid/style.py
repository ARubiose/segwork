from abc import ABC
from dataclasses import dataclass, field

DIAGRAM_TYPES = []
ORIENTATION_TYPES = ['TB', 'TD', 'BT', 'RL', 'LR']

#STYLES

#Shapes
ROUND_EDGES = ('(',')')
RHOMBUS = ('{', '}')
ASYMMETRIC_SHAPE = ('>', ']')

#Link
ARROW_LINK = ('-', '->')
DOTTED_LINK =('-.', '.->')




@dataclass
class MermaidStyle:
    diagram:str 
    orientation:str

@dataclass
class TorchFxStyle(MermaidStyle):   
    
    diagram:str = 'flowchart'
    orientation:str = field(default='TB')

    placeholder = {'shape':ROUND_EDGES}

    call_module = {'node':{'shape':ROUND_EDGES}, 
                   'link':{'shape':ARROW_LINK}}

    call_function = {'node':{'shape':RHOMBUS}, 
                     'link':{'shape':DOTTED_LINK}}
    
    output = {'node':{'shape':ASYMMETRIC_SHAPE},
              'link':{'shape':ARROW_LINK}}

    # get_attr = {'type':'', 'style':{}}
    # call_method = {'type':'', 'style':{}}





    
