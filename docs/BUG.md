# Known Bugs and Solution

## numpy ImportError

**20200613**

```text
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
ImportError: numpy.core.multiarray failed to import

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 968, in _find_and_load
SystemError: <class '_frozen_importlib._ModuleLockManager'> returned a result with an error set
ImportError: numpy.core._multiarray_umath failed to import
ImportError: numpy.core.umath failed to import
2020-06-13 08:57:38.485784: F tensorflow/python/lib/core/bfloat16.cc:675] Check failed: PyBfloat16_Type.tp_base != nullptr
```

Solution: upgrade your numpy version (1.18.5)
