""" Trainer class 

Template design pattern: https://refactoring.guru/es/design-patterns/template-method/python/example
The Template Method lets you turn a monolithic algorithm into a series of individual 
steps which can be easily extended by subclasses while keeping intact the structure defined in a superclass.
It’s okay if all the steps end up being abstract. However, some steps might benefit from having a default implementation. 
"""

from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    """Abstract base class that contains the skeleton of the training algorithm
    
    composed of calls to  abstract primitive operations.
        *   Data pipeline -> Function that returns a DataLoader
        *   Model building -> Function that returns a model
        *   Optimizer building -> Function that returns an optimizer
        *   Training process -> Function for updating the model (epoch function) -> Returns metrics
        *   Log results (params: epoch) -> Function to record metrics. Make use of logger

    Events? | Hook configuration files to methods
    
    """
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run trainer"""
        pass