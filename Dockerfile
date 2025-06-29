# Dockerfile for the Real-Time Fraud Detection API

# --- Base Image ---
# Start from an official Python base image. Using a specific version ensures consistency.
# The 'slim' variant is smaller, leading to a more efficient and secure image.
FROM python:3.9-slim

# --- Set Working Directory ---
# Define the working directory inside the container. All subsequent commands will be run from this path.
WORKDIR /app

# --- Copy Application Files ---
# Copy the requirements file first. This allows Docker to cache the installed packages
# unless the requirements file changes, speeding up subsequent builds.
COPY requirements.txt .

# --- Install Dependencies ---
# Install the Python dependencies specified in requirements.txt.
# The '--no-cache-dir' option reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code and model artifacts into the container.
COPY . .

# --- Expose Port ---
# Inform Docker that the container listens on port 8000 at runtime.
# This is the port Uvicorn will run on.
EXPOSE 8000

# --- Healthcheck (Optional but Recommended) ---
# Defines a command to check if the container is still working.
# This helps orchestration systems like Kubernetes manage the service's lifecycle.
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# --- Startup Command ---
# The command to run when the container starts.
# We use Uvicorn, a high-performance ASGI server, to run our FastAPI application.
# '0.0.0.0' makes the server accessible from outside the container.
# '--host' and '--port' bind the server to the specified network interface and port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
