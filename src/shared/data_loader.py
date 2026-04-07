"""
Shared data-loading utilities.

This file will later contain reusable dataset lookup logic so every
project follows the same loading order:
1. artifacts/processed
2. artifacts/raw
3. data/
4. optional external source
5. synthetic fallback
"""