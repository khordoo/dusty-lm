FROM python:3.11.15-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY tensor_worker.py /opt/tensor_worker.py
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
WORKDIR /work
ENTRYPOINT ["python", "-I", "/opt/tensor_worker.py"]
