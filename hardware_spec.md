# Hardware Specification

## Target Board
- **Model**: ARM Neoverse V3 development board (e.g., Ampere Altra or NXP i.MX‑8)
- **CPU**: Armv8.6‑a with SME and SVE2 extensions
- **Cores**: 8‑12 high‑performance cores
- **RAM**: 16 GB DDR4 (minimum 8 GB recommended)
- **Storage**: 64 GB eMMC (or micro‑SD) for OS and model data
- **Network**: Gigabit Ethernet

## Power & Cooling
- **Power Consumption**: ~15 W idle, up to 45 W under full load
- **Power Supply**: 12 V 2 A DC adapter (with optional UPS)
- **Cooling**: Passive heatsink with optional low‑profile fan (max 30 °C rise)

## OS
- **Linux Distribution**: Ubuntu 22.04 LTS (ARM64)
- **Kernel**: 5.15+ with SME/SVE2 support (default in Ubuntu 22.04)

## Connectivity
- **SSH**: Enabled for remote management
- **HTTP Port**: 8080 (configurable)

## Expansion
- **USB**: 2× USB‑C for power and optional external storage
- **GPIO**: Optional for status LEDs
