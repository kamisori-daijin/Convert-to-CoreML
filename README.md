# Convert-to-CoreML
Convert the model to coreml format

Coming soon

🚧 Note: I believe the main branch still lacks support for the PyTorch `__ior__` op.

As a result, models that utilize this op (such as gemma-3-1b-it) will fail to convert until a conversion function for `__ior__` is implemented.
