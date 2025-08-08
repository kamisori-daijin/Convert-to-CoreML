# Convert-to-CoreML
Convert the model to coreml format

Coming soon

🚧 Note: The RangeDim bug is expected to be fixed in `coremltools >= 8.3.1`.  
Currently, `coremltools 8.3.0` (available on PyPI) will crash when converting Gemma models due to an unimplemented `__ior__` operator in `RangeDim`.

**Temporary workaround:**  
Install the latest `coremltools` from the GitHub `main` branch until the fix is officially released:

```bash
pip uninstall coremltools
pip install git+https://github.com/apple/coremltools.git
