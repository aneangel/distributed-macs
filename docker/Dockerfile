FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev

# Set up C++ environment
WORKDIR /app/cpp
COPY cpp/CMakeLists.txt .
COPY cpp/src ./src
COPY cpp/binding.cpp .
RUN cmake . && make

# Set up Python environment
WORKDIR /app/python
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source
COPY python/src ./src
COPY python/main.py .

# Copy configs
COPY configs /app/configs

# Set working directory
WORKDIR /app

CMD ["python", "python/main.py"]