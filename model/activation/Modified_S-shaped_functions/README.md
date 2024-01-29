# Modified S-shaped Functions 

Zero-centered functions in s-shaped.



### Modified Sigmoid 

2 * Sigmoid(αx) - 1 

```python
import torch 
import torch.nn as nn 

class ModifiedSigmoid(nn.Module):
    def __init__(self, alpha=1.0):
        super(ModifiedSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))  # alpha 매개변수

    def forward(self, x):
        return 2*torch.sigmoid(self.alpha * x) - 1
```



### Modified Softsign

Softsign(αx)

```python
import torch 
import torch.nn as nn 

class ModifiedSoftsign(nn.Module):
    def __init__(self, alpha=1.0):
        super(ModifiedSoftsign, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))  # 가중치를 학습 가능한 매개변수로 설정

    def forward(self, x):
        x = self.alpha * x
        return x / (1 + torch.abs(x))
```



### Modified Tanh 

Tanh(αx)

```python
import torch 
import torch.nn as nn 

class ModifiedTanh(nn.Module):
    def __init__(self, alpha=0.5):
        super(ModifiedTanh, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))  # alpha 매개변수

    def forward(self, x):
        return torch.tanh(self.alpha * x)
```

