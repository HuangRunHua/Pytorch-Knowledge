################################################################################################################
# Tensors are a specialized data structure that are very similar to arrays and matrices. 
# In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
#
# Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. 
# In fact, tensors and NumPy arrays can often share the same underlying memory, 
# eliminating the need to copy data (see Bridge with NumPy). 
# Tensors are also optimized for automatic differentiation. 
# If you’re familiar with ndarrays, you’ll be right at home with the Tensor API.
################################################################################################################

import torch
import numpy as np

#########################################################################################################
# Initializing a Tensor
#   Tensors can be initialized in various ways.
#   [1] Directly from data.
#        Tensors can be created directly from data. The data type is automatically inferred.
#########################################################################################################
data = [[1, 2],
        [3, 4]]
x_data = torch.tensor(data)
"""
>>> tensor([[1, 2],
            [3, 4]])
"""

#########################################################################################################
#   [2] From a NumPy array
#       Tensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).
#########################################################################################################
np_array = np.array(data)
"""
>>> [[1 2]
     [3 4]]
"""
x_np = torch.from_numpy(np_array)

#########################################################################################################
#   [3] From another tensor:
#       The new tensor retains the properties (shape, datatype) of the argument tensor, 
#           unless explicitly overridden.
#########################################################################################################

# retains the properties of x_data
x_ones = torch.ones_like(x_data) 
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor: \n {x_rand} \n")
"""
>>> Ones Tensor: 
    tensor([[1, 1],
            [1, 1]]) 

>>> Random Tensor: 
    tensor([[0.4423, 0.3299],
            [0.9035, 0.9328]]) 
"""

#########################################################################################################
#   [4] With random or constant values:
#       shape is a tuple of tensor dimensions. In the functions below, 
#       it determines the dimensionality of the output tensor.
#########################################################################################################
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
"""
>>> Random Tensor: 
    tensor([[0.5289, 0.1512, 0.5405],
            [0.5642, 0.8397, 0.8233]]) 

>>> Ones Tensor: 
    tensor([[1., 1., 1.],
            [1., 1., 1.]]) 

>>> Zeros Tensor: 
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
"""

# Attributes of a Tensor
#   Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
"""
>>> Shape of tensor: torch.Size([3, 4])
>>> Datatype of tensor: torch.float32
>>> Device tensor is stored on: cpu
"""
