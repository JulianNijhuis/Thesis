# Use official PyTorch image with GPU runtime support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory inside container
WORKDIR /workspace

# Install system libraries required by OpenCV (libGL and GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /workspace

# Default command
CMD ["bash"]
