# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& rm -rf /var/lib/apt/lists/*

EXPOSE 8080

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set to user external login
ENV COP_EXTERN_LOGIN=1

# Install pip requirements, excluding 'ad' Git repo line
COPY requirements.txt .
RUN grep -v -i 'git+.*advancedanalytics.*/ad\.git' requirements.txt > temp_requirements.txt && \
    python -m pip install -r temp_requirements.txt && \
    rm temp_requirements.txt

# Install the GPX models
# Install the GPX models from the source distribution tarball
COPY gpx-zipped.tar.gz /tmp/gpx.tar.gz
RUN python -m pip install --no-cache-dir /tmp/gpx.tar.gz && rm -f /tmp/gpx.tar.gz

# Install the AD package from the vendored wheel
COPY ad-1.3.2-py3-none-any.whl /tmp/
RUN pip install /tmp/ad-1.3.2-py3-none-any.whl && rm /tmp/ad-1.3.2-py3-none-any.whl

WORKDIR /app
COPY . /app

# run the gpkit tests as the admin
RUN python -c "import gpkit"

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uwsgi", "--http-socket", "0.0.0.0:8080", "--enable-threads", "-w", "wsgi:app"]

