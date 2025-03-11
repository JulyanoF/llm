#!/bin/bash

/bin/ollama serve &

pid=$!

sleep 5

echo "🔴 Creating jbrsolutions model..."

ollama create jbrsolutions -f Modelfile
ollama pull jbrsolutions

echo "🟢 Done!"

wait $pid