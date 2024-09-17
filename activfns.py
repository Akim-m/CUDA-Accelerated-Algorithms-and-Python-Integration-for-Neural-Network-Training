import ctypes
import numpy as np

# Load the shared library
activation_lib = ctypes.CDLL('./activation_functions.so')

# Define argument and return types for activate function
activation_lib.activate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
activation_lib.activate.restype = None

# Define argument and return types for softmax function
activation_lib.softmax_activate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
activation_lib.softmax_activate.restype = None

# Python wrapper for the activation functions
def activate(input_array, activation_type):
    n = len(input_array)
    input_ctypes = (ctypes.c_float * n)(*input_array)
    output_ctypes = (ctypes.c_float * n)()

    activation_lib.activate(input_ctypes, output_ctypes, n, activation_type)
    
    return np.array(output_ctypes)

# Python wrapper for softmax
def softmax(input_array):
    n = len(input_array)
    input_ctypes = (ctypes.c_float * n)(*input_array)
    output_ctypes = (ctypes.c_float * n)()

    activation_lib.softmax_activate(input_ctypes, output_ctypes, n)
    
    return np.array(output_ctypes)

# Activation types mapping
activation_types = {
    'linear': 0,
    'sigmoid': 1,
    'tanh': 2,
    'relu': 3
}

# Example usage
if __name__ == "__main__":
    input_data = np.array([0.5, -1.0, 2.0, 0.0, -0.5], dtype=np.float32)
    
    # Apply Sigmoid activation
    output = activate(input_data, activation_types['sigmoid'])
    print("Sigmoid Output:", output)
    
    # Apply ReLU activation
    output = activate(input_data, activation_types['relu'])
    print("ReLU Output:", output)
    
    # Apply Softmax activation
    output = softmax(input_data)
    print("Softmax Output:", output)
