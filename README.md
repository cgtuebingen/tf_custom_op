Custom Op for TensorFlow
========================

**This will be deprecated soon. An [up-to-date version is here](https://github.com/PatWie/tensorflow_inference/tree/master/custom_op) without the need of compiling TensorFlow from source.**

This is a very simple example on adding custom C++/CUDA ops to TensorFlow and its intended usage is just being a starting point for other custom TensorFlow operations.

The current version is tested on TensorFlow v1.9. Run the script

```bash
cd user_ops
cmake .
make
python test_matrix_add.py
```
