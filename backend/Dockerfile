# syntax=docker/dockerfile:1.4.1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

ARG WORKDIR
ENV PATH=${WORKDIR}/vendor/bin:$PATH \
	PYTHONPATH=${WORKDIR}/vendor:${WORKDIR} \
	PYTHONUSERBASE=${WORKDIR}/vendor
WORKDIR ${WORKDIR}

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://ftp.jaist.ac.jp/pub/Linux/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://ftp.jaist.ac.jp/pub/Linux/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-venv \
    build-essential \
    git \
    curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Rust(for DeepFilterNet)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --target=${WORKDIR}/vendor -r requirements.txt

# === DeepFilterNet 部分をのみコピー ===
COPY src/models/DeepFilterNet-0.5.6 ${WORKDIR}/DeepFilterNet-0.5.6

# === libdfビルド ===
RUN pip install --no-cache-dir maturin && \
    cd ${WORKDIR}/DeepFilterNet-0.5.6 && \
    maturin build --release -m pyDF/Cargo.toml && \
	pip install $(find target/wheels -name "DeepFilterLib-*.whl") --target=${WORKDIR}/vendor

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as production

ENV DISPLAY=$DISPLAY
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージのインストール
COPY packages.txt /tmp/packages.txt
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://ftp.jaist.ac.jp/pub/Linux/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://ftp.jaist.ac.jp/pub/Linux/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-venv \
	vim \
	git \
	# 日本語環境
	task-japanese \
	locales \
	fonts-takao \
	task-japanese-desktop \
	ibus-kkc \
    # packages.txt
    && xargs -a /tmp/packages.txt apt-get install -y --no-install-recommends \
	# clean up
	&& apt-get autoremove -y \
	&& apt-get clean -y \
	&& rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python

# 日本語設定
RUN sed -i -E 's/# (ja_JP.UTF-8)/\1/' /etc/locale.gen \
	&& locale-gen
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
# JST
RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# セキュリティアップデートの適用
RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
		sudo \
		bsdutils \
		util-linux \
	&& apt-get autoremove -y \
	&& apt-get clean -y \
	&& rm -rf /var/lib/apt/lists/*

ARG WORKDIR
ENV PATH=${WORKDIR}/vendor/bin:$PATH \
	PYTHONPATH=${WORKDIR}/vendor:${WORKDIR} \
	PYTHONUSERBASE=${WORKDIR}/vendor
WORKDIR ${WORKDIR}

# ビルドステージからコピーした依存関係とアプリケーションのソースコードをコピー
COPY .cache/ ./.cache/
COPY --from=builder ${WORKDIR}/vendor ./vendor/
COPY src/ ./src/

# 環境変数の設定
# Pythonがpycファイルとdiscへの書き込みを行わないようにする
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# token
ARG HUGGINGFACE_TOKEN
ENV HUGGING_FACE_TOKEN=${HUGGINGFACE_TOKEN}
ENV HF_HUB_OFFLINE=1
ENV PYANNOTE_CACHE=${WORKDIR}/src/models/pyannote

# DeepFilterNetのバージョンを指定
ENV DEEPFILTERNET_VERSION=0.5.6
# DeepFilterNetのパスをPYTHONPATHに追加
ARG USER_HOME=/home/${USER}
ENV PYTHONPATH="${USER_HOME}/app/src/models/DeepFilterNet-0.5.6/DeepFilterNet:${PYTHONPATH}"

# アプリケーションの起動コマンド
CMD ["sleep", "infinity"]
