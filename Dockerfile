# ベースイメージとして python v3.6 を使用する
FROM python:3.8

# 以降の RUN, CMD コマンドで使われる作業ディレクトリを指定する
WORKDIR /app
RUN apt-get update
RUN apt-get -y upgrade
# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    tini \
    nfs-common \
    && apt-get clean

# Set fallback mount directory
ENV MNT_DIR /mnt/nfs/filestore
# カレントディレクトリにある資産をコンテナ上の ｢/app｣ ディレクトリにコピーする
ADD ./ /app

RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 8080
CMD ["streamlit","run","Neural_Network.py","--server.port","8080","--server.maxUploadSize=1028"]
