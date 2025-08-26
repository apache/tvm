# TVM Python Package Build System

This document describes the new Python package build system for Apache TVM, which uses `pyproject.toml` and `scikit-build-core` instead of the legacy `setup.py`.

## Overview

The new build system:
- Uses modern Python packaging standards (`pyproject.toml`)
- Integrates with CMake for C++ compilation
- Produces Python-version-agnostic wheels (`-py3-none-any.whl`)
- Provides better development experience with editable installs

## Prerequisites

- Python 3.8 or higher
- CMake 3.18 or higher
- C++ compiler (GCC, Clang, or MSVC)
- `scikit-build-core` >= 0.7.0

## Installation

### Development Install (Recommended for Development)

```bash
# Install in editable mode
pip install -e .

# This will:
# 1. Compile all C++ components using CMake
# 2. Install the package in editable mode
# 3. Make the `tvm` package importable
```

### Production Install

```bash
# Build and install from source
pip install .

# Or build a wheel first
pip wheel -w dist .
pip install dist/tvm-*.whl
```

## Building Wheels

### Local Wheel Build

```bash
# Build wheel for current platform
pip wheel -w dist .

# The wheel will be created in the `dist/` directory
# Format: tvm-0.16.0.dev0-py3-none-linux_x86_64.whl
```

### Cross-Platform Wheel Build

For building wheels for multiple platforms, you can use tools like `cibuildwheel`:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for multiple platforms
cibuildwheel --platform linux --arch x86_64 .
```

## Configuration

### CMake Options

The build system passes these CMake options by default:
- `-DTVM_FFI_ATTACH_DEBUG_SYMBOLS=ON`
- `-DTVM_FFI_BUILD_TESTS=OFF`
- `-DTVM_FFI_BUILD_PYTHON_MODULE=ON`
- `-DTVM_BUILD_PYTHON_MODULE=ON`

### Custom CMake Options

You can override CMake options by setting environment variables:

```bash
# Example: Enable CUDA support
export CMAKE_ARGS="-DUSE_CUDA=ON -DUSE_CUTLASS=ON"
pip install -e .
```

## Package Structure

The built package includes:
- Python source files (`tvm/`)
- Compiled shared libraries (`libtvm.so`, `libtvm_runtime.so`)
- Third-party dependencies (CUTLASS, FlashAttention, etc.)
- Header files and CMake configuration
- Documentation and licenses

## Development Workflow

1. **Clone the repository**
   ```bash
   git clone https://github.com/apache/tvm.git
   cd tvm
   ```

2. **Install in editable mode**
   ```bash
   pip install -e .
   ```

3. **Make changes to Python code**
   - Changes are immediately available without reinstallation

4. **Make changes to C++ code**
   - Rebuild is required: `pip install -e . --force-reinstall`

5. **Test your changes**
   ```bash
   python test_installation.py
   ```

## Troubleshooting

### Common Issues

1. **CMake not found**
   ```bash
   # Install CMake
   pip install cmake
   # Or use system package manager
   sudo apt install cmake  # Ubuntu/Debian
   brew install cmake      # macOS
   ```

2. **Compiler not found**
   - Ensure you have a C++ compiler installed
   - On Windows, install Visual Studio Build Tools
   - On macOS, install Xcode Command Line Tools

3. **Build fails with CUDA**
   - Ensure CUDA toolkit is properly installed
   - Set `CMAKE_ARGS="-DUSE_CUDA=ON"`
   - Check CUDA version compatibility

4. **Import error after installation**
   ```bash
   # Check if package is installed
   pip list | grep tvm
   
   # Try reinstalling
   pip install -e . --force-reinstall
   ```

### Debug Build

For debugging build issues:

```bash
# Enable verbose output
export SKBUILD_VERBOSE=1

# Install with debug info
pip install -e . --verbose
```

## Migration from setup.py

The old `setup.py` workflow has been replaced:

| Old (setup.py) | New (pyproject.toml) |
|----------------|----------------------|
| `python setup.py install` | `pip install .` |
| `python setup.py develop` | `pip install -e .` |
| `python setup.py bdist_wheel` | `pip wheel -w dist .` |
| `python setup.py sdist` | `pip install build && python -m build` |

## Contributing

When contributing to TVM:

1. **Use the new build system** - Don't modify `setup.py`
2. **Test your changes** - Run `python test_installation.py`
3. **Update dependencies** - Modify `pyproject.toml` if needed
4. **Follow Python packaging best practices**

## References

- [PEP 518](https://www.python.org/dev/peps/pep-0518/) - Specifying Build System Requirements
- [PEP 517](https://www.python.org/dev/peps/pep-0517/) - A build-system independent format for source trees
- [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/)
- [TVM FFI build system](ffi/pyproject.toml) - Reference implementation
