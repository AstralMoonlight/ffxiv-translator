# AstralES — Guía completa de traducción masiva con Vast.ai

## PARTE 1 — Preparación local (PC Windows)

### 1.1 Ajustar .gitignore

Edita `D:\Proyectos\ffxiv-translator\.gitignore`:

```gitignore
# Inputs: pesados, no se suben
input/

# Temporales
checkpoints/
logs/
__pycache__/
*.pyc
```

> `output/` ya NO está ignorado — se subirá al repo.

---

### 1.2 Copiar glossary.json actual

Copia tu `glossary.json` completo (el del proyecto AstralES-Data) a:
```
D:\Proyectos\ffxiv-translator\glossary.json
```

---

### 1.3 Crear requirements.txt

Crea `D:\Proyectos\ffxiv-translator\requirements.txt`:

```
questionary
requests
```

---

### 1.4 Crear setup.sh (para Vast.ai)

Crea `D:\Proyectos\ffxiv-translator\setup.sh`:

```bash
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
sleep 3

# Descargar modelo
echo "[3/4] Descargando qwen2.5:14b-instruct-q4_K_M (~9GB)..."
ollama pull qwen2.5:14b-instruct-q4_K_M

# Instalar dependencias Python
echo "[4/4] Instalando dependencias Python..."
pip install -r requirements.txt -q

echo ""
echo "=== Setup completo. Listo para traducir. ==="
echo ""
echo "Ejecuta: python translate.py --no-menu"
```

---

### 1.5 Inicializar el repositorio local

Abre PowerShell en `D:\Proyectos\ffxiv-translator` y ejecuta:

```powershell
git init
git remote add origin git@github.com:AstralMoonlight/ffxiv-translator.git
git branch -M main
git add .
git commit -m "init: estructura base del traductor"
git push -u origin main
```

> Asegúrate de que el repo esté creado como **privado** en GitHub antes de hacer push.

---

## PARTE 2 — Configuración de Vast.ai

### 2.1 Crear cuenta y cargar saldo

1. Ve a https://vast.ai y crea una cuenta
2. Ve a **Billing** → **Add Credit**
3. Carga al menos **$5 USD** (alcanza de sobra para el MSQ completo)

---

### 2.2 Agregar clave SSH

Para conectarte a la instancia necesitas una clave SSH:

1. En tu PC, abre PowerShell:
```powershell
# Verificar si ya tienes una clave
cat ~\.ssh\id_rsa.pub

# Si no existe, generarla
ssh-keygen -t rsa -b 4096
```

2. Copia el contenido de `~\.ssh\id_rsa.pub`
3. En Vast.ai ve a **Account** → **SSH Keys** → pega la clave

---

### 2.3 Rentar una instancia RTX 3090

1. Ve a **Search** en Vast.ai
2. Filtra:
   - **GPU:** RTX 3090
   - **Min VRAM:** 24 GB
   - Ordena por **precio ascendente**
3. Busca una instancia con:
   - Precio ~$0.07–$0.10/hr
   - Disco: mínimo **30 GB** (9GB modelo + inputs + outputs)
4. En **Image**, selecciona:
   ```
   pytorch/pytorch:latest
   ```
   o busca `ollama` si hay imagen disponible
5. En **On-start script**, puedes dejarlo vacío
6. Click **Rent** → confirmar

> Espera 1-2 minutos hasta que el estado sea **Running**

---

### 2.4 Conectarse por SSH

En el panel de Vast.ai, una vez la instancia esté **Running**, verás un comando SSH como:

```bash
ssh -p 12345 root@123.45.67.89
```

Ejecútalo desde PowerShell en tu PC.

---

## PARTE 3 — Trabajo en Vast.ai

### 3.1 Clonar el repositorio

Una vez conectado por SSH:

```bash
# Configurar git
git config --global user.email "tu@email.com"
git config --global user.name "AstralMoonlight"

# Clonar el repo
git clone https://github.com/AstralMoonlight/ffxiv-translator.git
cd ffxiv-translator
```

> Como el repo es privado necesitas autenticarte. La forma más simple es
> usar un **Personal Access Token** de GitHub:
> GitHub → Settings → Developer Settings → Personal Access Tokens → Generate new token
> Luego clona con:
> ```bash
> git clone https://tu-token@github.com/AstralMoonlight/ffxiv-translator.git
> ```

---

### 3.2 Subir los inputs

Los inputs no están en el repo (están en .gitignore).
Tienes dos opciones:

**Opción A — scp desde tu PC** (en PowerShell local, no en SSH):
```powershell
scp -P 12345 -r D:\Proyectos\ffxiv-translator\input root@123.45.67.89:/root/ffxiv-translator/
```

**Opción B — subir un .zip a Google Drive y descargarlo en la instancia:**
```bash
# En la instancia Vast.ai
pip install gdown -q
gdown "https://drive.google.com/uc?id=TU_ID_DE_ARCHIVO"
unzip inputs.zip -d input/
```

---

### 3.3 Ejecutar el setup

```bash
chmod +x setup.sh
./setup.sh
```

Esto instala Ollama, descarga el modelo (~9GB, tarda ~5 minutos) e instala dependencias.

---

### 3.4 Verificar que Ollama funciona

```bash
ollama list
# Debe mostrar: qwen2.5:14b-instruct-q4_K_M

# Test rápido
ollama run qwen2.5:14b-instruct-q4_K_M "Traduce al español: Speak with Alphinaud."
```

---

### 3.5 Lanzar la traducción

```bash
# Modo headless (recomendado para sesiones largas)
python translate.py --no-menu

# O con nohup para que siga corriendo si se cierra el SSH
nohup python translate.py --no-menu > traduccion.log 2>&1 &

# Ver el progreso en tiempo real
tail -f traduccion.log
```

> Con `nohup` puedes cerrar el SSH y la traducción sigue corriendo en la instancia.
> Para reconectarte luego, simplemente vuelve a hacer SSH.

---

### 3.6 Hacer push de los outputs (durante o al final)

Puedes hacer push parcial cada cierto tiempo como backup:

```bash
cd ffxiv-translator
git add output/
git commit -m "traduccion: msq/01_arr_2_0 completo"
git push
```

O al terminar todo:

```bash
git add output/
git commit -m "traduccion: catalogo completo"
git push
```

---

## PARTE 4 — Finalizar

### 4.1 Descargar outputs a tu PC (opcional si ya hiciste push)

```powershell
# En PowerShell local
scp -P 12345 -r root@123.45.67.89:/root/ffxiv-translator/output D:\Proyectos\ffxiv-translator\
```

---

### 4.2 Destruir la instancia

**MUY IMPORTANTE:** Una vez terminado, ve a Vast.ai → **Instances** → **Destroy**.
Si no destruyes la instancia, sigues pagando por hora.

---

### 4.3 Revisión de anomalías

Los archivos en `logs/` contienen las líneas sospechosas detectadas durante la traducción:

```powershell
# Ver anomalías del MSQ
cat D:\Proyectos\ffxiv-translator\logs\msq\01_arr_2_0\all_texts_en_anomalies.jsonl
```

Cada línea es un JSON con:
- `key` — la key problemática
- `en` — texto original en inglés
- `es` — lo que generó el modelo
- `reason` — por qué fue marcada (`identical_to_source`, `batch_failed_fallback_individual`)

---

## Resumen de comandos críticos

| Acción | Comando |
|---|---|
| Setup Vast.ai | `./setup.sh` |
| Traducir todo | `python translate.py --no-menu` |
| Ver progreso | `tail -f traduccion.log` |
| Backup outputs | `git add output/ && git commit -m "backup" && git push` |
| Detener traducción | `Ctrl+C` (el checkpoint se guarda automáticamente) |
| **Destruir instancia** | Vast.ai web → Destroy |

