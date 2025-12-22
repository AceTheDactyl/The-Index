# PlatformIO Projects

Hardware and firmware projects for physical computing and IoT devices using PlatformIO development environment.

## Overview

This folder contains embedded systems projects, hardware schematics, and firmware implementations for various physical computing applications.

## Projects

### [RHZ Stylus (ESP32-S3)](./PlatformIO-main/)
Active stylus firmware project using the ESP32-S3 microcontroller.

**Features:**
- OPT3001 ambient light sensor integration
- Custom PCB design
- PlatformIO build configuration

## Directory Structure

```
PlatformIO-main/
└── PlatformIO-main/
    ├── hardware/         # PCB designs and schematics
    ├── firmware/         # Microcontroller code
    │   └── stylus_maker_esp32s3/
    │       ├── src/      # Source code
    │       └── lib/      # Libraries (Adafruit_OPT3001)
    ├── docs/             # Documentation
    ├── packages/         # Package definitions
    └── templates/        # Project templates
```

## Development

Projects use [PlatformIO](https://platformio.org/) for embedded development. Install PlatformIO IDE or CLI to build and flash firmware.

```bash
# Build firmware
pio run

# Upload to device
pio run --target upload

# Monitor serial output
pio device monitor
```

---

*Part of [Ace Systems Examples](../index.html)*
