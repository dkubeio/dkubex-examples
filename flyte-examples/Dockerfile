# Use the specified base image
FROM cr.flyte.org/flyteorg/flytekit:py3.10-1.9.0
USER root
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip3 --version
RUN pip --version
# Install opencv-python
RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install tflite
RUN pip install optuna
RUN pip install mlflow
RUN pip install ray
USER flytekit
# Set any environment variables if needed
# ENV MY_VARIABLE=value

# Define the entry point for the container
# ENTRYPOINT ["/path/to/entry_script.sh"]

# Specify any working directory if needed
# WORKDIR /app

# Copy any additional files if needed
# COPY my_file.py /app/

# Set any additional commands or arguments for the entry point
# CMD ["my_script.py"]
