FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
# Healthcheck so compose `depends_on: condition: service_healthy` works.
HEALTHCHECK --interval=3s --timeout=2s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/health').status==200 else 1)"
EXPOSE 9000
CMD ["python", "app.py"]
