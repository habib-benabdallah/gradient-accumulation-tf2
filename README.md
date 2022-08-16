# gradient-accumulation-tf2
Wrapper objects that allow gradient accumulation during training in Tensorflow 2.

The repository requires Numpy and Tensorflow 2 to be used. To enable gradient accumulation in your code, start by importing the BatchOptimizer wrapper from gradient_accumulation.py.

```python
from gradient_accumulation import BatchOptimizer
```

Then, you need to define your model and to define your training data size before proceeding. When you are about to use an optimizer, Adam in this example, wrap it inside the BatchOptimizer object as such.

```python
# Use gradient accumulation for 
optimizer
```
