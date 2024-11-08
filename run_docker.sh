#!/bin/bash

echo "killing old docker processes"
docker compose rm -fs

echo "building docker containers"
docker compose build && \
docker compose up --detach && \
docker compose logs --follow
