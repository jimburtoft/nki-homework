#!/usr/bin/env bash
# Gradescope autograder setup (runs once when the image is built).
# Gradescope's base image is Ubuntu 22.04; this script installs Python 3.12, libarchive
# (needed by the Neuron compiler), and the NKI toolchain.
set -euo pipefail
apt-get update -qq
apt-get install -y -qq software-properties-common >/dev/null
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv libarchive13 >/dev/null
python3.12 -m venv /opt/nkivenv
/opt/nkivenv/bin/pip install -q --index-url https://pip.repos.neuron.amazonaws.com \
    --extra-index-url https://pypi.org/simple "nki==0.7.0" "neuronx-cc==2.28.*" numpy ml_dtypes
/opt/nkivenv/bin/python /autograder/source/common/selftest.py
