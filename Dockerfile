FROM python:3.8-slim

RUN mkdir /code
WORKDIR /code

RUN pip install -U pip
# install project dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

ENTRYPOINT ["python3.8", "main.py"]