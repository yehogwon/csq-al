import numpy as np
import torch

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC

class CSVM: 
    C = 10
    GAMMA = 1
    KERNEL = 'sigmoid'

    def __init__(self, n_classes: int=10, seed: int=42, verbose=False): 
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.initialize()
    
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]: 
        probs = self.svc.predict_proba(x)
        full_probs = torch.zeros((x.shape[0], self.n_classes), dtype=torch.float32)
        present_classes = self.svc.classes_
        present_classes = torch.tensor(present_classes, dtype=torch.int64)
        full_probs[:, present_classes] = torch.tensor(probs, dtype=torch.float32)
        return full_probs, None
    
    def initialize(self): 
        self.svc = SVC(C=self.C, gamma=self.GAMMA, kernel=self.KERNEL, probability=True, random_state=self.seed, verbose=self.verbose)
    
    def apply(self, *args, **kwargs): 
        print('Note: calling apply from CSVM directly initializes the classifier.')
        self.initialize()

    def fit(self, x: np.ndarray, y: np.ndarray): 
        self.svc.fit(x, y)
    
    def train(self, *args, **kwargs): 
        return self

    def eval(self, *args, **kwargs): 
        return self

    def to(self, *args, **kwargs): 
        return self
