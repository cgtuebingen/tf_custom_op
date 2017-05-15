import tensorflow as tf
import numpy as np
from matrix_add import matrix_add


class MatrixAddtest(tf.test.TestCase):

    def test_forward(self):
        matA = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10
        matB = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10
        bias = 42.

        expected = matA + matB + bias

        tensorA = tf.Variable(matA, dtype=tf.float32)
        tensorB = tf.Variable(matB, dtype=tf.float32)

        actual = matrix_add(tensorA, tensorB, bias)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(actual)

        self.assertTrue(np.all(actual == expected))

    def test_backward(self):
        matA = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10
        matB = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10
        bias = 42.

        expected = (matA + matB + bias).astype(np.float32)

        tensorA = tf.Variable(matA, dtype=tf.float32)
        tensorB = tf.Variable(matB, dtype=tf.float32)

        actual = matrix_add(tensorA, tensorB, bias)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            err = tf.test.compute_gradient_error(
                [tensorA, tensorB], [matA.shape, matB.shape],
                actual, expected.shape)

        self.assertLess(err, 1e-2)

if __name__ == '__main__':
    tf.test.main()
