FROM python:3.11-slim

# Update package index, Install curl + C compiler for Torch Inductor, delete package index
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*                                            

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install deps
COPY pyproject.toml uv.lock /app/                         
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --frozen --no-install-project     

# Copy app and install project
COPY . /app
RUN uv sync --frozen  

EXPOSE 8000
# App startup
CMD ["python", "-m", "dreamsculpt_be.main"]