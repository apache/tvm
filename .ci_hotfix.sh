#!/bin/bash
# Hotfix for GPU CI ml-dtypes conflict
echo "Applying ml-dtypes hotfix..."
pip install "ml-dtypes>=0.4.0,<0.5.0" --force-reinstall
