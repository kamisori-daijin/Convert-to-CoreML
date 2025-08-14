# Convert-to-CoreML
Convert the model to coreml format

🚧 Note: I believe the main branch still lacks support for the PyTorch `__ior__` op.

As a result, models that utilize this op (such as gemma-3-1b-it) will fail to convert until a conversion function for `__ior__` is implemented.

**How  To Use**
If the model you want to convert is in  `/path/your/directory/`:

```bash
python -m convert_to_coreml --model "/path/your/directory/"
```
or
```bash
python -m convert_to_coreml --model="/path/your/directory/"
```


**Credits**

[coremltools][] -Library for conversion

[coremltools]: url "https://github.com/apple/coremltools"







