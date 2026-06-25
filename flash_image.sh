#!/usr/bin/env bash
# flash_image.sh - Flash Ubuntu 22.04 LTS onto the ARM board
# Usage: sudo ./flash_image.sh /dev/sdX

set -e

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <block_device>"
  exit 1
fi

DEVICE=$1

echo "Flashing Ubuntu 22.04 LTS to $DEVICE..."

# Download Ubuntu image (placeholder URL)
IMAGE_URL="https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04-preinstalled-server-arm64+raspi.img.xz"
TMP_IMG="/tmp/ubuntu-22.04.img"

curl -L $IMAGE_URL -o /tmp/ubuntu-22.04.img.xz
xz -d /tmp/ubuntu-22.04.img.xz -c > $TMP_IMG

# Write image to device
sudo dd if=$TMP_IMG of=$DEVICE bs=4M status=progress conv=fdatasync

sync

echo "Flashing completed. You can now boot the board."
