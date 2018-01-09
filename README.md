Custom Op for TensorFlow
========================

This is a very simple example on adding custom C++ ops to TensorFlow and its intended usage is just being a starting point for other custom TensorFlow operations.

The current version is tested on [4cb0c13](https://github.com/tensorflow/tensorflow/commit/4cb0c13c7779da536cac6c682180c5757611b384). When building TF from source using the gcc5 you will need to use cmake by

```
cmake . -DUSE_NEW_ABI=ON
```

to match `ABI=1` Note, the official packages are built with `ABI=0`.

*workaround:* until [15002](https://github.com/tensorflow/tensorflow/issues/15002) is fixed, you need to have access to the TF git repository, like

```bash
git clone /path/to/tf
export TensorFlow_GIT_DIR=/path/to/tf
```