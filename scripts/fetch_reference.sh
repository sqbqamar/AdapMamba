#!/usr/bin/env bash
# Fetch the VMamba v0 reference (as shipped with VM-UNet) for tests/test_equivalence.py.
set -euo pipefail
curl -sL -o vmunet_vmamba.py https://raw.githubusercontent.com/JCruan519/VM-UNet/main/models/vmunet/vmamba.py
pip install einops timm >/dev/null
python tests/test_equivalence.py vmunet_vmamba.py
