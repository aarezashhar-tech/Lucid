#!/usr/bin/env bash
echo "Starting Lucid AI News Verification Platform..."

trap 'kill 0' EXIT

# Start backend
cd backend
pip install -r requirements.txt
python app.py &
BACKEND_PID=$!
cd ..

# Start frontend
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

wait
