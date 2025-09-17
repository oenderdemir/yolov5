FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    x11vnc \
    git \
    xvfb \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ultralytics ve opencv yükle
RUN pip install --no-cache-dir ultralytics yolov5 opencv-python

COPY .. .

ENV DISPLAY=:0
EXPOSE 5900

CMD bash -c "Xvfb :0 -screen 0 1280x720x16 & fluxbox & \
    x11vnc -display :0 -nopw -forever -quiet & \
    python plakaBulucu.py"
