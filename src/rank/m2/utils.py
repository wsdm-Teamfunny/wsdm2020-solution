import sys
import time
import numpy as np
from tqdm import tqdm
import torch
from matchzoo.engine.base_metric import sort_and_couple, RankingMetric


def build_matrix(term_index, gv_model, dim=256):
    
    input_dim = len(term_index)
    matrix = np.empty((input_dim, dim))

    valid_keys = gv_model.vocab.keys()
    for term, index in term_index.items():
        if term in valid_keys:
            matrix[index] = gv_model.word_vec(term)
        else:
            if '<unk>' in gv_model.vocab.keys():
                matrix[index] = gv_model.word_vec("<unk>")
            else:
                matrix[index] = np.random.randn(dim).astype(dtype=np.float32)
    return matrix

def topk_lines(df, k):
    print(df.shape)
    df.loc[:, 'rank'] = df.groupby(['description_id', 'type']).cumcount().values
    df = df[df['rank'] < k]
    df.drop(['rank'], axis=1, inplace=True)
    print(df.shape)
    return df


class MAP(RankingMetric):
    
    def __init__(self, k = 3):
        self._k = k
        
    def __repr__(self) -> str:
        return 'mean_average_precision@{}'.format(self._k)
    
    def __call__(self, y_true, y_pred):
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            if idx+1>self._k:
                return 0
            if label > 0:
                return 1. / (idx + 1)
        return 0.
    

def predict(trainer, testloader):
    with torch.no_grad():
        trainer._model.eval()
        predictions = []
        for batch in tqdm(testloader):
            inputs = batch[0]
            outputs = trainer._model(inputs).detach().cpu()
            predictions.append(outputs)
        trainer._model.train()

    return torch.cat(predictions, dim=0).numpy()


class Logger:
    def __init__(self, log_filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(log_filename, "a")
        self.log.write("="*10+" Start Time:"+time.ctime()+" "+"="*10+"\n")
    
    def __enter__(self):
        sys.stdout = self
 
    def __exit__(self, e_t, e_v, t_b):
        sys.stdout = self.close()
        
    def stop_log(self):
        sys.stdout = self.close()
        
    def write(self, message):
        self.terminal.write(message)
        if message=="\n":
            self.log.write(message)
        else:
            self.log.write("["+time.ctime()+"]: "+message)
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.write("="*10+" End Time"+time.ctime()+" "+"="*10+"\n")
        self.log.close()
        return self.terminal

