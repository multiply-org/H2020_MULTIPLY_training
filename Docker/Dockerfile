# Start from the mambaorg/micromamba base image
FROM mambaorg/micromamba:latest


# Copy the environment.yml file to the container
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install --freeze-installed -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && rm -rf /opt/conda/pkgs/*


ARG MAMBA_DOCKERFILE_ACTIVATE=1 


# Copy the Jupyter notebooks to the container
COPY . .

# Expose port 8888 for Jupyter notebook
EXPOSE 8888

# Start Jupyter notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
