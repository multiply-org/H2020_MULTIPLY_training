# Environment build stage
# Using miniconda image to build the required image
FROM condaforge/mambaforge:latest AS build
# Install conda-pack
RUN mamba install conda-pack && mamba config --set remote_read_timeout_secs 600
# Create the conda environment using environment.yml file
COPY environment.yml .
RUN mamba env create -f environment.yml
# Package the environment as tar using conda-pack
RUN conda-pack -n multiply -o /tmp/env.tar && \
 mkdir /venv && cd /venv && tar -xf /tmp/env.tar && \
 rm /tmp/env.tar
# Unpack the environment in /venv using conda-unpack
RUN /venv/bin/conda-unpack
# Runtime stage
# using slim debian image as the base image for the environment
FROM debian:bullseye-slim AS runtime
# Copy the environment directory /venv from the build stage:
COPY --from=build /venv /venv
COPY . /tmp/multiply

EXPOSE 8888
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
           cd /tmp/multiply && \
	   jupyter lab --no-browser --ip=0.0.0.0 --port=8888 \
           --allow-root --NotebookApp.token=''
