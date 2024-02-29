



From gcr.io/deeplearning-platform-release/tf2-gpu.2-8

WORKDIR /

COPY trainer /trainer
COPY requirements.txt /


RUN pip install -r requirements.txt



ENTRYPOINT [ "python" ,"-m", "trainer.final"]