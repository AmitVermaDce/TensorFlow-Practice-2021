import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignoring the GPU error
import tensorflow as tf

# print(tf.__version__)  # Tensorflow version

# <------------Basics------------------>
x = tf.constant(2)  # Initializing a random constant

y = tf.constant(2, shape=(2, 1), dtype=tf.int32)  # Initialization constant with shape and type

simple_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3),
                            dtype=tf.int32)  # Initialization a matrix with shape and size

one_matrix = tf.ones((3, 3))  # Initialization of One's Matrix

zeros_matrix = tf.zeros((2, 2))  # Initialization of Zero's matrix

eye_matrix = tf.eye(3)  # Initialization of Identity Matrix

matrix_of_normal_distribution = tf.random.normal((3, 3), mean=0,
                                                 stddev=1)  # Initialization of normaly distributed matrix
matrix_of_uniform_distribution = tf.random.uniform((3, 3), minval=0,
                                                   maxval=1)  # Initialization of uniformly distributed matrix

vec = tf.range(9)  # A simple vector with 9 elements started with 0 and ends at 8

vec_x = tf.range(start=3, limit=13, delta=2)  # vector with start, end and stepsize

vec_dtype = tf.cast(vec, dtype=tf.float64)  # casting of elements from one type to another

# <------------Mathematical Operations------------------>

xx = tf.constant([1, 2, 3, 4])
yy = tf.constant([11, 22, 33, 44])
xx_add_yy = tf.add(xx, yy)  # Element wise addition: xx + yy
xx_subtract_yy = tf.subtract(xx, yy)  # Element wise subtraction: xx - yy
xx_divide_yy = tf.divide(xx, yy)  # Element wise-division: xx/yy
xx_multiply_yy = tf.multiply(xx, yy)  # Element wise-multiplication: xx*yy
xx_dot_product_yy = tf.tensordot(xx, yy,
                                 axes=1)  # Dot product: Element-wise multiplication(xx1*yy1) then summation of all elements
xx_dot_product_yy_2 = tf.reduce_sum(xx * yy,
                                    axis=0)  # Dot product: Element-wise multiplication(xx1*yy1) then summation of all elements
xx_scale = xx ** 5  # Scales or multiply each element with a value of 5
matrix1 = tf.random.normal((2, 3))
matrix2 = tf.random.normal((3, 2))
matrix1_cross_product_matrix2 = tf.matmul(matrix1, matrix2)  # Matrix multiplication of 2 matrix
matrix1_cross_product_matrix2_2 = matrix1 @ matrix2  # Matrix Multiplication using only @symbol

# <------------Indexing------------------>
xi = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
# print(xi[:])  # print entire elements from matrix
# print(xi[1:6])
# print(xi[::2])  # Step size of 2 prints 0,2,4.. elements
# print(xi[::-1])  # Print elements in reverse order
indices = tf.constant([0, 3])
x_ind = tf.gather(xi, indices)  # Only elements with specific indices

x_2d = tf.constant([[1, 2], [3, 4], [5, 6]])
# print(x_2d[0, :])  # Only 1st row and all elements from the same column
# print(x_2d[0:2, :]) # Only 1st and 2nd row and all elements from columns

# <------------Reshaping------------------>
xxx = tf.range(9)
xx_reshape = tf.reshape(xxx, (3, 3))  # reshaping of vectors
xx_transpose = tf.transpose(xx_reshape, perm=[1, 0])  # Transpose of matrix
