FROM nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20

COPY run.py /opt/isaac-sweep/run.py

ENTRYPOINT ["/isaac-sim/python.sh", "/opt/isaac-sweep/run.py"]
