FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ADD . /bot
WORKDIR /bot
RUN apt update && apt install -y ffmpeg && apt install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install -U setuptools pip
RUN pip install setuptools-rust
RUN pip install -r requirements.txt --no-cache-dir --quiet
CMD python bot.py