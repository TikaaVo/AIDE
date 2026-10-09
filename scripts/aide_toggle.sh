#!/bin/bash
# Show/hide the AIDE window. Bind this to a global keyboard shortcut.
echo "toggle" | socat - UNIX-CONNECT:/tmp/aide.sock
