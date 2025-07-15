#!/bin/bash

set -eu

user="$(whoami)"
uid="$(groups ${user} | awk '{print $3}')"
sudo chown -R "${user}:${uid}" "/home/${user}"

exec "$@"