#!/bin/bash
set -e

# Clean up build products not located in _build/:
rm -rf dev_docs/api/generated/

# Clean up build products with `make clean`:
make clean
