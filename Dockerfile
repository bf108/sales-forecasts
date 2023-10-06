FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.10 as build

# Install Python dependencies
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
