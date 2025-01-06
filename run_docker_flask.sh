#!/bin/bash

echo "killing old docker processes"
docker compose -f docker-compose-flask.yml rm -fs

echo "building docker containers"
docker compose -f docker-compose-flask.yml build && \
docker compose -f docker-compose-flask.yml up --detach && \
docker compose -f docker-compose-flask.yml logs --follow
