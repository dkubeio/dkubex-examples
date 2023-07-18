# ---- Base Node ----
FROM nikolaik/python-nodejs:python3.10-nodejs18 AS base

ARG APPDIR
ARG APP_URLPREFIX

RUN test -n "$APPDIR" || (echo "APPDIR  must be set" && false)

WORKDIR /app
COPY securechat/ .
#COPY $APPDIR/data .
COPY $APPDIR/requirements.txt .
COPY ./pkgs/* ./pkgs/
COPY $APPDIR/*.py ./pages/api/
RUN npm i

RUN pip install -r requirements.txt
RUN pip install pkgs/*
EXPOSE 3000

CMD ["npm", "run", "dev"]
