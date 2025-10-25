# poke-ckks

Homomorphic similarity search using the CKKS scheme via Pyfhel. The project encrypts a catalog of vectors, computes dot products between an incoming query vector and each encrypted entry, and returns the top-k most similar vectors without exposing raw data during computation.

## Features

- CKKS context bootstrap with configurable polynomial degree, scale, and modulus chain.
- Encryption helper for vector catalogs using `Pyfhel`.
- Homomorphic element-wise multiplication between a plaintext query and encrypted catalog entries.
- Decryption and summation of element-wise products to compute dot product similarity scores.
- Ranking utilities to select the top-k matches once results are authorized to be revealed.
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

- The element-wise product is computed homomorphically (on encrypted data), preserving privacy during the multiplication step. However, the final summation to compute the dot product happens after decryption. In a deployment scenario, this decryption step should occur in a trusted environment that holds the secret key.
- For fully homomorphic dot product computation (including the summation), rotation-based slot aggregation can be implemented, but requires careful handling of CKKS slot rotations and scale management.
- CKKS is approximate; dot products are subject to scale-dependent rounding. Adjust the scale and modulus chain to meet precision requirements.
