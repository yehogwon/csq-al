import numpy as np
import torch
from .strategy import Strategy

from torch.utils.data import Subset

class EntropySampling(Strategy):
	def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
		super(EntropySampling, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict(Subset(self.train_raw_dataset, idxs_unlabeled))
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]
