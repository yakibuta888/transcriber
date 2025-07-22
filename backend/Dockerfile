# syntax=docker/dockerfile:1.4.1
FROM python:3.12.11-slim as builder

ARG WORKDIR
ENV PATH=${WORKDIR}/vendor/bin:$PATH \
	PYTHONPATH=${WORKDIR}/vendor:${WORKDIR} \
	PYTHONUSERBASE=${WORKDIR}/vendor
WORKDIR ${WORKDIR}

# Rust(for DeepFilterNet)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --target=${WORKDIR}/vendor -r requirements.txt


FROM python:3.12.11-slim as production

ENV DISPLAY=$DISPLAY
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージのインストール
COPY packages.txt /tmp/packages.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# 日本語設定
RUN sed -i -E 's/# (ja_JP.UTF-8)/\1/' /etc/locale.gen \
	&& locale-gen
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

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

# Create the user
ARG USERNAME
ARG HOMEDIR
ARG USERID
ARG GROUPID
RUN <<-EOF
	echo "Create User = ${USERID}. Group = ${GROUPID}"
	groupadd -g ${GROUPID} ${USERNAME}
	useradd -m -s /bin/bash -d ${HOMEDIR} -u ${USERID} -g ${GROUPID} ${USERNAME}
	mkdir -p /etc/sudoers.d/
	echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME}
	chmod 0440 /etc/sudoers.d/${USERNAME}
    chown -R ${USERID}:${GROUPID} ${HOMEDIR}
EOF
USER ${USERNAME}

# ビルドステージからコピーした依存関係とアプリケーションのソースコードをコピー
COPY --chown=${USERID}:${GROUPID} src/ ./src/
COPY --chown=${USERID}:${GROUPID} --from=builder ${WORKDIR}/vendor ./vendor/

# 環境変数の設定
# Pythonがpycファイルとdiscへの書き込みを行わないようにする
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# アプリケーションの起動コマンド
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
# CMD ["python", "src/main.py"]
CMD ["sleep", "infinity"]
