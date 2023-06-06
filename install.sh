#!/bin/bash

# Wheel is never depended on, but always needed. MulticoreTSNE requires lower CMake version
pip install wheel cmake==3.18.4

pip install -e .
