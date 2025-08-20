import numpy as np
import torch
import importlib
from functools import partial

from sklearn.calibration import CalibratedClassifierCV

class CSVM: 
    C = 10
    GAMMA = 1
    KERNEL = 'sigmoid'

    def __init__(self, n_classes: int=10, seed: int=42, verbose=False, svm_gpu=False): 
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.svm_gpu = svm_gpu

        if self.svm_gpu:
            _cuml_svm_module = importlib.import_module('cuml.svm')
            self.svc_cls = getattr(_cuml_svm_module, 'SVC')
            self.svc_constructor = partial(
                self.svc_cls,
                probability=False
            )
        else:
            _sklearn_svm_module = importlib.import_module('sklearn.svm')
            self.svc_cls = getattr(_sklearn_svm_module, 'SVC')
            self.svc_constructor = partial(
                self.svc_cls,
                probability=True
            )

        self.initialize()
    
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]: 
        try:
            probs = self.svc.predict_proba(x)
        except AttributeError:
            print(f'SVC {type(self.svc)} does not support predict_probs')
            margins_np = self.svc.decision_function(x)
            margins = torch.from_numpy(margins_np)
            probs = torch.softmax(margins, dim=1)

        full_probs = torch.zeros((x.shape[0], self.n_classes), dtype=torch.float32)
        present_classes = self.svc.classes_
        present_classes = torch.tensor(present_classes, dtype=torch.int64)
        full_probs[:, present_classes] = torch.tensor(probs, dtype=torch.float32)
        return full_probs, None
    
    def initialize(self): 
        self.svc = self.svc_constructor(C=self.C, gamma=self.GAMMA, kernel=self.KERNEL, random_state=self.seed, verbose=self.verbose)
    
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
