# Start from the mambaorg/micromamba base image
FROM mambaorg/micromamba:latest

# Set the working directory to /app

# Copy the environment.yml file to the container
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1 


# Copy the Jupyter notebooks to the container
COPY . .

# Expose port 8888 for Jupyter notebook
EXPOSE 8888

# Start Jupyter notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
