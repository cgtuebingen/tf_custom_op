import tensorflow as tf
import numpy as np
from matrix_add import matrix_add


class MatrixAddtest(tf.test.TestCase):

    def _forward(self, use_gpu=False, dtype=np.float32):
        matA = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        matB = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        bias = 42.

        expected = matA + matB + bias
        actual_op = matrix_add(matA, matB, bias)

        with self.test_session(use_gpu=use_gpu) as sess:
            actual = sess.run(actual_op)

        self.assertShapeEqual(expected, actual_op)
        self.assertAllClose(expected, actual)

    def test_forward_float(self):
        self._forward(use_gpu=False, dtype=np.float32)
        self._forward(use_gpu=True, dtype=np.float32)

    def test_forward_double(self):
        self._forward(use_gpu=False, dtype=np.float64)
        self._forward(use_gpu=True, dtype=np.float64)

    def _backward(self, use_gpu=False, dtype=np.float32):
        matA = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        matB = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        bias = 42.

        expected = (matA + matB + bias).astype(np.float32)

        matA_op = tf.convert_to_tensor(matA)
        matB_op = tf.convert_to_tensor(matB)

        actual_op = matrix_add(matA_op, matB_op, bias)

        with self.test_session():
            err = tf.test.compute_gradient_error(
                [matA_op, matB_op], [matA.shape, matB.shape],
                actual_op, expected.shape)

        self.assertLess(err, 1e-2)

    def test_backward_float(self):
        self._backward(use_gpu=False, dtype=np.float32)
        self._backward(use_gpu=True, dtype=np.float32)

    def test_backward_double(self):
        self._backward(use_gpu=False, dtype=np.float64)
        self._backward(use_gpu=True, dtype=np.float64)


if __name__ == '__main__':
    tf.test.main()
