# ByToken

**ByToken** is a fast and minimal Byte Pair Encoding (BPE) tokenizer written entirely in C++ for high performance and exposed to Python using `pybind11`. This allows you to train, encode, and decode text efficiently, making it suitable for machine learning preprocessing pipelines.


## Features

- Written in modern C++ for speed
- Custom training from raw text corpus with custom vocab size
- Efficient encoding and decoding
- Exposed to Python using `pybind11`
- Easy integration with Python projects


## Installation

Either install from pip directly via `pip install bytoken` or build from source (below).

### 1. Clone the Repository

```bash
git clone https://github.com/sly-sloth/nanogpt-self.git
cd bytoken
```

### 2. Build the C++ extension

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### 3. Extract the Python Extension Module
The compiled `.so` or (`.pyd` in Windows) file will be placed in the `build/` directory. You may add/replace it in the `bytoken/` subfolder (non-repo) and then continue to use it by importing. Refer the code below or the file : `bytoken_test.py` for clarity.

```python
from bytoken.bytoken import ByToken

tokenizer = ByToken()
tokenizer.train("hello world", vocab_size=100, verbose=True)

encoded = tokenizer.encode("hello world")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
```

## Performance Benchmark

The C++ implementation of the ByToken tokenizer offers a significant performance boost over the pure Python version. Below is a summary of performance comparisons on identical tasks.

### Test Setup

- **Corpus size**: ~440,000 characters
- **Iterations**: 50,000 encode/decode cycles on a short paragraph (~1000 chars)
- **System**: 6-core CPU, 16 GB RAM (Fedora Linux)
- **Tokenizer vocab size**: 256
- **Python Version**: 3.13  
- **Built with**: CMake, pybind11, and `-O3` optimization flag

### Timing Results

| Operation         | Python Time | C++ Time (.so) |
|------------------|-------------|----------------|
| Training          | 12.84 s     | 2.18 s         |
| 50,000 Iterations | 115.50 s    | 13.47 s        |
| Speedup           | —           | ~9-10x faster     |

### Summary

This performance difference highlights how wrapping C++ implementations using `pybind11` can dramatically improve execution time for compute-heavy tasks like tokenization. Ideal for production-level language model preprocessing.
