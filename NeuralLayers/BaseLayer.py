import numpy as np
import pandas as pd

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forwardProp(self, input):
        pass
    
    def backwardProp(self, outputGradient, learningRate):
        pass
    
    