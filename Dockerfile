FROM python:3.11-slim

# Install curl
RUN apt update
RUN apt-get install -y curl

# Torch Inductor requires a C compiler
RUN apt-get update && apt-get install -y build-essential

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy app to working directory and install deps
ADD . /app
WORKDIR /app
# uv doesnt allow you to directly install lockfile dependencies to the system env. Need to first export to requirements.txt,
#   and use the pip interface. uv's interface does not respect UV_SYSTEM_PYTHON. This is annoying, I'm being forced to maintain 2 lockfiles now.
RUN uv pip install -r requirements.txt --system

EXPOSE 8000
# App startup
CMD ["python", "-m", "dreamsculpt_be.main"]