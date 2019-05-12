# SGP

Simple gaussian process regression

## Installation

`pip install sgp`

## Getting started

```python
from sgp.variational import VariationalGP as VGP
import numpy as np

X_train = np.linspace(0, 5, 1000)
y_train = np.sin(X_train) * X_train

model = VGP().fit(X_train.reshape(-1, 1), y_train)
X_test = np.linspace(0, 10, 100)
y_test, std_test = model.predict(X_test.reshape(-1, 1), return_std=True)
```


