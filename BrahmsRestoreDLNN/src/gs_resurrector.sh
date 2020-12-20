#!/bin/bash
  
until python dlnn_brahms_restore.py g TRUE -k; do
        echo "GS Python script crashed with exit code $?. Respawning..." >&2
        sleep 5
done
