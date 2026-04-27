#!/bin/bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
export LSL_FORCE_DEFAULT_MULTICAST=0
python3 lsl_server.py --camera 1 --filter kalman