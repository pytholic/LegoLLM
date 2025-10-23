"""Create a dummy .bin file for testing.

Created by @pytholic on 2025.10.23
"""

# We can read tiny shakespeare .bin file and use first 1000 tokens to create a dummy .bin file for testing.

from pathlib import Path

import numpy as np

# read tiny shakespeare .bin file
tiny_shakespeare_bin_path = Path("data/processed/tiny_shakespeare/train.bin")
tiny_shakespeare_bin = np.fromfile(tiny_shakespeare_bin_path, dtype=np.uint16)

# use first 1000 tokens to create a dummy .bin file for testing
dummy_bin = tiny_shakespeare_bin[:1000]
dummy_bin.tofile("data/processed/tiny_shakespeare/dummy_train.bin")
