# openzl4zarr
OpenZL codec for Zarr

This Repository is a proof of concept. It provides a Zarr codec and an example.

## Installation

Clone this git repository together with its submodule `git clone --recursive`

### OpenZL

Enter the `openzl4zarr/openzl` repository and follow the instructions to build the library.

Enter the `openzl4zarr/openzl/py` directory and install the Python extension.

```bash
pip install .
```

### Library

Change the directory back to `openzl4zarr` and install the Python package using

```bash
pip install .[example]
```
