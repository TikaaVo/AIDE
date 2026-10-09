#!/bin/bash
# Start/stop voice recording. Bind this to a global keyboard shortcut.
echo "record_toggle" | socat - UNIX-CONNECT:/tmp/aide.sock
