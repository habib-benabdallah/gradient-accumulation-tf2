# gradient-accumulation-tf2
 Gradient accumulation allows memory-bound systems to circumvent their memory limitation at the cost of computational resources. Sometimes, a small batch size leads to a noisy training, and a bigger one is needed. However, bigger batch sizes require more memory, and can be impossible to handle on certain systems. This repository allows you to achieve a smooth training with big batch sizes without the need for more memory. It contains wrapper objects that allow gradient accumulation during training in Tensorflow 2.

The repository requires Numpy and Tensorflow 2 to be used. To enable gradient accumulation in your code, start by importing the BatchOptimizer wrapper from gradient_accumulation.py.

```python
from gradient_accumulation import BatchOptimizer
```

Then, you need to define your model and to define your training set size before proceeding. When you are about to use an optimizer, Adam in this example, wrap it inside the BatchOptimizer object.

```python
# Define your model
model = ...
# Determine the size of your training set
train_size = ...
# Determine the maximum batch size that is supported by your system. We suppose that it is 16 here
maximum_batch_size = 16
# Choose the batch size that is optimal for your training but not supported by your system. We suppose that it is 256 here
desired_batch_size = 256
# Choose a tensorflow optimizer 
optimizer = ...
# Wrap this optimizer inside a BatchOptimizer object
batch_optimizer = BatchOptimizer(optimizer, maximum_batch_size, desired_batch_size, train_size, model)
# Compile your model with the batch optimizer
model.compile(optimizer=batch_optimizer, ...)
# Start the training using a batch size equal to maximum_batch_size
model.fit(x, y, batch_size=maximum_batch_size, ...)
```

You can run `test.py` to check the difference in precision between using gradient accumulation and not using it on the MNIST dataset. For a better training approximation using gradient accumulation, use 'float64' weights in your model. To do so, add the following line before defining your model.

```python
tensorflow.keras.backend.set_floatx('float64')
```

It is also recommended to define the maximum batch size and the desired batch size as powers of 2. Doing so will improve the gradient accumulation precision since their use will be equivalent to using shift operations on the gradients.
