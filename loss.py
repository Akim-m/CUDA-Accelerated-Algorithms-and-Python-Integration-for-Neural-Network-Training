import ctypes
import numpy as np

# Load the shared library
loss_lib = ctypes.CDLL('./loss_functions.so')

# Define argument and return types for loss functions
loss_lib.mae_loss.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
loss_lib.mae_loss.restype = None

loss_lib.mse_loss.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
loss_lib.mse_loss.restype = None

loss_lib.cross_entropy_loss_host.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
loss_lib.cross_entropy_loss_host.restype = None

loss_lib.huber_loss_host.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
loss_lib.huber_loss_host.restype = None

# Python wrapper for loss functions
def mae_loss(y_true, y_pred):
    n = len(y_true)
    y_true_ctypes = (ctypes.c_float * n)(*y_true)
    y_pred_ctypes = (ctypes.c_float * n)(*y_pred)
    result = ctypes.c_float()
    
    loss_lib.mae_loss(y_true_ctypes, y_pred_ctypes, n, ctypes.byref(result))
    
    return result.value

def mse_loss(y_true, y_pred):
    n = len(y_true)
    y_true_ctypes = (ctypes.c_float * n)(*y_true)
    y_pred_ctypes = (ctypes.c_float * n)(*y_pred)
    result = ctypes.c_float()
    
    loss_lib.mse_loss(y_true_ctypes, y_pred_ctypes, n, ctypes.byref(result))
    
    return result.value

def cross_entropy_loss(y_true, y_pred):
    n = len(y_true)
    y_true_ctypes = (ctypes.c_float * n)(*y_true)
    y_pred_ctypes = (ctypes.c_float * n)(*y_pred)
    result = ctypes.c_float()
    
    loss_lib.cross_entropy_loss_host(y_true_ctypes, y_pred_ctypes, n, ctypes.byref(result))
    
    return result.value

def huber_loss(y_true, y_pred, delta=1.0):
    n = len(y_true)
    y_true_ctypes = (ctypes.c_float * n)(*y_true)
    y_pred_ctypes = (ctypes.c_float * n)(*y_pred)
    result = ctypes.c_float()
    
    loss_lib.huber_loss_host(y_true_ctypes, y_pred_ctypes, n, delta, ctypes.byref(result))
    
    return result.value

# Example usage
if __name__ == "__main__":
    y_true = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.4], dtype=np.float32)
    
    # Compute MAE
    print("MAE Loss:", mae_loss(y_true, y_pred))
    
    # Compute MSE
    print("MSE Loss:", mse_loss(y_true, y_pred))
    
    # Compute Cross-Entropy Loss
    print("Cross-Entropy Loss:", cross_entropy_loss(y_true, y_pred))
    
    # Compute Huber Loss
    print("Huber Loss:", huber_loss(y_true, y_pred))
