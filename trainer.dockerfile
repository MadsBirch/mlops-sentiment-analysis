# Base image
FROM python:3.10-slim

# Install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy over our application (the essential parts) from our computer to the container:
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY test_environment.py test_environment.py
COPY src/ src/
COPY data/ data/
COPY Makefile Makefile

# Installs git into the Docker image:
RUN apt-get update && apt-get install git -y

# To use make: 
RUN apt install build-essential -y --no-install-recommends

# Set the working directory in our container and add commands that install the dependencies:
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Retrieve the data: 
RUN make data

# Remove the extra src folder that gets created: 
RUN rm -r ./src/src
RUN pip install -e .
#RUN wandb.login(key="bd3fd38b22f78a0420f42abfc8b978d7ae49d44c")

# Name our training script as the entrypoint (CMD) for our docker image. The entrypoint is the application that we want to run when the image is being executed:
CMD ["make", "train"]