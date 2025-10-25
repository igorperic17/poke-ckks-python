# poke-ckks

Homomorphic similarity search using the CKKS scheme via Pyfhel. The project encrypts a catalog of vectors, computes dot products between an incoming query vector and each encrypted entry, and returns the top-k most similar vectors without exposing raw data during computation.

## Features

- CKKS context bootstrap with configurable polynomial degree, scale, and modulus chain.
- Encryption helper for vector catalogs using `Pyfhel`.
- Fully homomorphic dot product computation using element-wise multiplication and rotation-based slot summation.
- Ranking utilities to decrypt similarity scores and select the top-k matches.
- Pytest-based regression test demonstrating correctness against NumPy.

## Getting Started

### Prerequisites

- Python 3.9+
- System dependencies required by [Pyfhel](https://github.com/ibarrond/Pyfhel#requirements) (Microsoft SEAL backend). On Ubuntu you typically need `build-essential`, `cmake`, and `libboost-all-dev`.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .[test]
```

### Running the Demo

```bash
python examples/run_demo.py
```

The script encrypts a small catalog, evaluates a query vector homomorphically, decrypts the scores, and prints the top matches.

### Running Tests

```bash
pytest
```

## Project Layout

```
src/poke_ckks/       Core library code
examples/            Usage examples
tests/               Automated tests
```

## Security Notes

- The entire dot product computation (element-wise multiplication AND summation) is performed homomorphically on encrypted data, preserving privacy throughout the computation.
- Only the final dot product scores are decrypted in the trusted environment that holds the secret key.
- Top-k selection currently decrypts the encrypted scores for ranking. In a deployment scenario, perform this step in a trusted environment, or extend the logic with secure comparison protocols if fully private ranking is required.
- CKKS is approximate; dot products are subject to scale-dependent rounding. Adjust the scale and modulus chain to meet precision requirements.
