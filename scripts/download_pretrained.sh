#!/usr/bin/env bash
# ImageNet-1K VMamba-Small (v0) checkpoint from the official VMamba release; the same
# weights VM-UNet initialises from. Stage depths [2, 2, 27, 2]; AdapMamba-UNet uses
# stages 1, 2 and 4 in full and the first three blocks of stage 3.
set -euo pipefail
mkdir -p pretrained
URL="https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmsmall_dp03_ckpt_epoch_238.pth"
curl -L -o pretrained/vssmsmall_dp03_ckpt_epoch_238.pth "$URL"
sha256sum pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
