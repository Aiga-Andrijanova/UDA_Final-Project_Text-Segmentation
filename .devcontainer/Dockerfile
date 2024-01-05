FROM mcr.microsoft.com/vscode/devcontainers/python:3.11-bookworm

RUN pip3 install torch torchvision torchsummary
RUN pip3 install opencv-python
RUN apt update -y && apt install -y ca-certificates fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgbm1 libgcc1 libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 lsb-release wget xdg-utils
RUN pip3 install torchmetrics pandas numpy tqdm tensorflow==2.15.0 pdf2image poppler-utils torchsummary albumentations

# Install microsoft fonts
RUN sed -i '/Components: main/s/$/ contrib/' /etc/apt/sources.list.d/debian.sources && apt update && apt install -y ttf-mscorefonts-installer
RUN apt update -y && apt install -y fonts-takao fonts-indic fonts-noto fonts-noto-cjk

# Install mc
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /usr/local/minio-binaries/mc
RUN chmod -R a+x /usr/local/minio-binaries &&\
    chown 1000:1000 /usr/local/minio-binaries
ENV PATH=$PATH:/usr/local/minio-binaries/
