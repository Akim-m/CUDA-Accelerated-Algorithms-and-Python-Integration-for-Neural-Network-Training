import ctypes
import numpy as np

# Load the shared libraries
normal_matrix_lib = ctypes.CDLL('./normal_matrix_multiplication.so')
cublas_matrix_lib = ctypes.CDLL('./cublas_matrix_multiplication.so')

# Define argument and return types for both functions
normal_matrix_lib.normal_matrix_multiply.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
normal_matrix_lib.normal_matrix_multiply.restype = None

cublas_matrix_lib.cublas_matrix_multiply.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.POINTER(ctypes.c_float), 
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
cublas_matrix_lib.cublas_matrix_multiply.restype = None

# Wrapper for normal matrix multiplication
def normal_matrix_multiply(A, B):
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    
    m, k = A.shape
    k_, n = B.shape
    assert k == k_, "Incompatible dimensions for matrix multiplication."
    
    C = np.zeros((m, n), dtype=np.float32)
    
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    normal_matrix_lib.normal_matrix_multiply(A_ptr, B_ptr, C_ptr, m, n, k)
    return C

# Wrapper for cuBLAS-based matrix multiplication
def cublas_matrix_multiply(A, B):
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    
    m, k = A.shape
    k_, n = B.shape
    assert k == k_, "Incompatible dimensions for matrix multiplication."
    
    C = np.zeros((m, n), dtype=np.float32)
    
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    cublas_matrix_lib.cublas_matrix_multiply(A_ptr, B_ptr, C_ptr, m, n, k)
    return C

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    
    print("Normal Matrix Multiplication Result:")
    C = normal_matrix_multiply(A, B)
    print(C)
    
    print("\ncuBLAS Matrix Multiplication Result:")
    C = cublas_matrix_multiply(A, B)
    print(C)
