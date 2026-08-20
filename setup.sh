#!/bin/bash
set -e

echo "=== AstralES Vast.ai Setup ==="

# Instalar Ollama si no está
if ! command -v ollama &> /dev/null; then
    echo "[1/4] Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/4] Ollama ya instalado."
fi

# Iniciar servidor Ollama en background
echo "[2/4] Iniciando Ollama..."
ollama serve &> /tmp/ollama.log &

# Esperar hasta que Ollama responda (máx 30 segundos)
echo "      Esperando que Ollama esté listo..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "      OK (${i}s)"
        break
    fi
    sleep 1
done

# Descargar modelo
echo "[3/4] Descargando qwen2.5:14b-instruct-q4_K_M (~9GB)..."
ollama pull qwen2.5:14b-instruct-q4_K_M

# Instalar dependencias Python
echo "[4/4] Instalando dependencias Python..."
pip install -r requirements.txt -q

echo ""
echo "=== Setup completo. Listo para traducir. ==="
echo ""
echo "Ejecuta: nohup python translate.py --no-menu > traduccion.log 2>&1 &"