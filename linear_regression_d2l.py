import torch
from d2l import torch as d2l
from data import SyntheticDataLinReg

class LinearRegression(d2l.Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super(LinearRegression).__init__()
        self.save_hyperparameters()
        self.w=torch.normal(0,sigma, (num_inputs,1), requires_grad=True)
        self.b=torch.zeros(1, requires_grad=True)

def add_to_class(Class):
    def wrapper(object):
        setattr(Class, object.__name__, object)
    return wrapper

@d2l.add_to_class(LinearRegression)
def forward(self,x):
    return torch.matmul(x, self.w)+self.b

@d2l.add_to_class(LinearRegression)
def loss(self, y, y_hat):
    l=(y-y_hat)**2/2
    return l.mean()

class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        super(SGD).__init__()
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            param-=self.lr*param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@d2l.add_to_class(LinearRegression)
def configure_optimizer(self):
    return SGD([self.w, self.b],self.lr)

@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss=self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val>0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx+=1
    if self.val_dataloader is None:
        return 
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx+=1

w=torch.tensor([2, -3], dtype=torch.float32)
b=torch.tensor(4, dtype=torch.float32)
data=d2l.SyntheticRegressionData(w,b)
model=LinearRegression(2, lr=0.001)
trainer=d2l.Trainer(max_epochs=5)
trainer.fit(model=model, data=data)