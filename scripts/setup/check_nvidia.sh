#!/bin/bash
nvidia-smi
nvcc --version 2>/dev/null || echo "nvcc 未找到"
