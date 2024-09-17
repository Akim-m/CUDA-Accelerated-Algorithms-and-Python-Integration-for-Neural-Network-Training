import ctypes
import numpy as np

# Load the shared library
opt_lib = ctypes.CDLL('./optimizers.so')

# Define argument and return types for each optimizer
opt_lib.sgd_optimizer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int]
opt_lib.sgd_optimizer.restype = None

opt_lib.adam_optimizer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                   ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                   ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, 
                                   ctypes.c_int, ctypes.c_int]
opt_lib.adam_optimizer.restype = None

opt_lib.rmsprop_optimizer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_float, 
                                      ctypes.c_float, ctypes.c_float, ctypes.c_int]
opt_lib.rmsprop_optimizer.restype = None

# Python wrapper for SGD
def sgd_optimizer(weights, gradients, lr=0.01):
    n = len(weights)
    weights_ctypes = (ctypes.c_float * n)(*weights)
    gradients_ctypes = (ctypes.c_float * n)(*gradients)
    
    opt_lib.sgd_optimizer(weights_ctypes, gradients_ctypes, ctypes.c_float(lr), ctypes.c_int(n))
    
    return np.array(weights_ctypes)

# Python wrapper for Adam
def adam_optimizer(weights, gradients, m, v, beta1=0.9, beta2=0.999, alpha=0.001, epsilon=1e-8, t=1):
    n = len(weights)
    weights_ctypes = (ctypes.c_float * n)(*weights)
    gradients_ctypes = (ctypes.c_float * n)(*gradients)
    m_ctypes = (ctypes.c_float * n)(*m)
    v_ctypes = (ctypes.c_float * n)(*v)
    
    opt_lib.adam_optimizer(weights_ctypes, gradients_ctypes, m_ctypes, v_ctypes, 
                           ctypes.c_float(beta1), ctypes.c_float(beta2), ctypes.c_float(alpha), 
                           ctypes.c_float(epsilon), ctypes.c_int(t), ctypes.c_int(n))
    
    return np.array(weights_ctypes), np.array(m_ctypes), np.array(v_ctypes)

# Python wrapper for RMSprop
def rmsprop_optimizer(weights, gradients, s, alpha=0.001, beta=0.9, epsilon=1e-8):
    n = len(weights)
    weights_ctypes = (ctypes.c_float * n)(*weights)
    gradients_ctypes = (ctypes.c_float * n)(*gradients)
    s_ctypes = (ctypes.c_float * n)(*s)
    
    opt_lib.rmsprop_optimizer(weights_ctypes, gradients_ctypes, s_ctypes, 
                              ctypes.c_float(alpha), ctypes.c_float(beta), ctypes.c_float(epsilon), ctypes.c_int(n))
    
    return np.array(weights_ctypes), np.array(s_ctypes)

# Example usage
if __name__ == "__main__":
    # Initializing weights, gradients, and moments with some random values
    weights = np.array([0.5, -0.3, 0.8, 0.7], dtype=np.float32)
    gradients = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    
    # SGD example
    print("SGD:")
    updated_weights_sgd = sgd_optimizer(weights, gradients, lr=0.01)
    print("Updated weights:", updated_weights_sgd)
    
    # Adam example
    m = np.zeros_like(weights, dtype=np.float32)
    v = np.zeros_like(weights, dtype=np.float32)
    print("\nAdam:")
    updated_weights_adam, m_adam, v_adam = adam_optimizer(weights, gradients, m, v, t=2)
    print("Updated weights:", updated_weights_adam)
    print("Updated m:", m_adam)
    print("Updated v:", v_adam)
    
    # RMSprop example
    s = np.zeros_like(weights, dtype=np.float32)
    print("\nRMSprop:")
    updated_weights_rmsprop, s_rmsprop = rmsprop_optimizer(weights, gradients, s)
    print("Updated weights:", updated_weights_rmsprop)
    print("Updated s:", s_rmsprop)
