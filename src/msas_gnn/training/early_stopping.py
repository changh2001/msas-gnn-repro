"""验证集早停（patience=20，监控验证集准确率）。"""
import logging; logger = logging.getLogger(__name__)
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta; self.best=0.0; self.counter=0; self.should_stop=False
    def step(self, val_acc):
        if val_acc > self.best+self.min_delta: self.best=val_acc; self.counter=0
        else:
            self.counter+=1
            if self.counter>=self.patience: logger.info(f"早停 best={self.best:.4f}"); self.should_stop=True
        return self.should_stop
