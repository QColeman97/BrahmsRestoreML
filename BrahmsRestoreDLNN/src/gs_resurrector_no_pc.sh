#!/bin/bash
  
until python dlnn_brahms_restore.py g FALSE -k; do
        echo "GS Python script crashed with exit code $?. Respawning..." >&2
        sleep 120
done
