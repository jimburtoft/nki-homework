# CPU-only environment for the NKI homework: simulator + Neuron compiler.
#   docker build -t nki-homework .
#   docker run --rm -it -v "$PWD":/hw -w /hw nki-homework python common/selftest.py
FROM python:3.12-slim

# libarchive is the only system library the Neuron compiler back end needs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libarchive13 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      --index-url https://pip.repos.neuron.amazonaws.com \
      --extra-index-url https://pypi.org/simple \
      "nki==0.7.0" "neuronx-cc==2.28.*" numpy ml_dtypes

WORKDIR /hw
CMD ["python", "common/selftest.py"]
