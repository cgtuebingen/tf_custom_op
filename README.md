Custom Op for TensorFlow
========================

This is a very simple example on adding custom C++ ops to TensorFlow and its intended usage is just being a starting point for other custom TensorFlow operations.

The current version is tested on [4cb0c13](https://github.com/tensorflow/tensorflow/commit/4cb0c13c7779da536cac6c682180c5757611b384). Run the script

```
python configure.py
```

to match find all necessary paths and settings.

*workaround:* until [15002](https://github.com/tensorflow/tensorflow/issues/15002) is fixed and in a release version, you need to have access to the TF git repository, like

```bash
git clone /path/to/tf
export TensorFlow_GIT_DIR=/path/to/tf
```