import json
import os
import re
import time
import urllib.request
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# Config desde .env
# ─────────────────────────────────────────────

def load_config() -> dict:
    config = {
        "OLLAMA_URL":       "http://localhost:11434",
        "MODEL_NAME":       "qwen2.5:14b-instruct-q4_K_M",
        "BATCH_SIZE":       15,
        "CHECKPOINT_EVERY": 10,
        "TEMPERATURE":      0,  # Temperatura baja para acatar reglas, pero sin romper gramática
    }
    env_path = Path(__file__).parent / "config.env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    config["BATCH_SIZE"]       = int(config["BATCH_SIZE"])
    config["CHECKPOINT_EVERY"] = int(config["CHECKPOINT_EVERY"])
    config["TEMPERATURE"]      = float(config["TEMPERATURE"])
    return config

# ─────────────────────────────────────────────
# Detección de tipo
# ─────────────────────────────────────────────

def detect_entry_type(key: str) -> str:
    k = key.upper()
    if "_SEQ_"  in k or "_JRNL_" in k: return "journal"
    if "_TODO_" in k:                   return "duty_list"
    return "dialogues"

def detect_file_type(filename: str) -> str | None:
    name = Path(filename).name.lower()
    if "action"         in name: return "actions"
    if "status_effect"  in name: return "status_effects"
    if "trait"          in name: return "traits"
    if "ui_string"      in name: return "ui"
    if "menus_addon"    in name: return "ui"
    if "tutorial"       in name: return "ui"
    if "system_message" in name: return "ui"
    return None

# ─────────────────────────────────────────────
# Glosario Dinámico
# ─────────────────────────────────────────────

def load_glossary(glossary_path: Path) -> dict:
    """Carga el glosario como diccionario en lugar de texto plano."""
    if not glossary_path.is_file():
        return {}
    try:
        return json.loads(glossary_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [Error] Glosario: {e}")
        return {}

def get_relevant_glossary(text_batch: list[str], glossary_data: dict) -> str:
    """Filtra el glosario para inyectar solo los términos presentes en el batch actual."""
    if not glossary_data:
        return "No hay términos específicos para este bloque."
        
    combined_text = " ".join(str(t) for t in text_batch).lower()
    lines = []
    
    for category, mapping in glossary_data.items():
        for src, tgt in mapping.items():
            if src.lower() in combined_text:
                # Instrucción positiva absoluta (hack para que no traduzca)
                lines.append(f'- "{src}" → "{tgt}"')
                    
    if not lines:
        return "No hay términos del glosario en este bloque. Traduce normalmente."
    return "\n".join(lines)

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

def build_proper_noun_fence(glossary_data: dict) -> str:
    """
    Construye una lista explícita de sustantivos propios que el modelo NO debe tocar.
    Se extraen las categorías que representan nombres, lugares, facciones y entidades
    cuyo valor en el glosario es idéntico a la clave (= se mantienen sin traducir).
    """
    protected_categories = {
        "characters", "factions_and_organizations",
        "key_locations_and_dungeons", "races_and_entities",
        "game_commands_and_emotes",
    }
    names: list[str] = []
    for cat, mapping in glossary_data.items():
        if cat not in protected_categories:
            continue
        for src, tgt in mapping.items():
            # Solo incluir los que se mantienen igual (no hay traducción especial)
            if src.strip() == tgt.strip():
                names.append(src)
    if not names:
        return ""
    # Limitar la lista para no saturar el contexto; los más cortos son los más peligrosos
    names_sorted = sorted(names, key=len)
    sample = names_sorted[:120]
    return (
        "SUSTANTIVOS PROPIOS INTOCABLES (copia el texto original, no traduzcas ni adaptes):\n"
        + ", ".join(f'"{n}"' for n in sample)
    )


def build_system_prompt(file_type: str, relevant_glossary: str, proper_noun_fence: str = "") -> str:
    proper_block = f"\n\n{proper_noun_fence}" if proper_noun_fence else ""

    base = f"""GLOSARIO OBLIGATORIO PARA ESTE BLOQUE (reemplazo exacto, sin excepción):
{relevant_glossary}
{proper_block}

REGLAS DE LOCALIZACIÓN — JERARQUÍA ESTRICTA:
1. GLOSARIO PRIMERO: Si una palabra aparece en el glosario, usa la traducción del glosario. Sin variaciones, sin paráfrasis.
2. SUSTANTIVOS PROPIOS: Nombres de personajes, lugares, facciones, razas y términos del juego que NO aparecen en el glosario se copian en inglés tal como están. NUNCA los inventes ni los traduzcas.
3. TEXTO TRADUCIBLE: Solo traduce el texto narrativo/descriptivo que no sea un sustantivo propio.
4. COMANDOS: Nunca traduzcas comandos que empiecen con / (ej: /wave, /stretch).
5. VARIABLES: Preserva {"{0}"}, {"{1}"}, %s, <nombre>, <clase> y cualquier placeholder exactamente igual.
6. FORMATO: Devuelve ÚNICAMENTE las líneas traducidas en el formato numerado indicado. Sin explicaciones, notas ni texto extra.
7. ANTI-INVENCIÓN: Si no estás seguro del significado de un término propio, mantenlo en inglés. No adivines."""

    specific = {
        "duty_list":      "TIPO: Objetivo de misión (Duty List)\n- Imperativo conciso: \"Speak with\" → \"Habla con\", \"Find\" → \"Busca a\", \"Head to\" → \"Dirígete a\".\n- Breve y directo.",
        "journal":        "TIPO: Diario de misión (Journal)\n- Tono narrativo y fluido, como una crónica de aventuras.\n- Adapta expresiones idiomáticas al español neutro.",
        "dialogues":      "TIPO: Diálogo de personaje\n- Preserva el tono y registro de cada personaje (formal/informal, arcaico, rústico).\n- Doblaje natural de fantasía épica. Respeta puntuación original (elipsis, exclamaciones, tartamudeos).",
        "actions":        "TIPO: Acción de combate\n- Nombres: cortos e impactantes. Descripciones: claras en términos de mecánica de juego.",
        "status_effects": "TIPO: Efecto de estado\n- Nombres concisos (1-3 palabras). HP/MP se mantienen igual.",
        "traits":         "TIPO: Rasgo de clase\n- Nombres descriptivos y coherentes.",
        "ui":             "TIPO: Interfaz de usuario (UI)\n- Texto extremadamente conciso. Preserva todas las variables y placeholders.",
    }

    block = specific.get(file_type, "TIPO: Texto general de videojuego.")
    return (
        "Eres un locutor de videojuegos especializado en Final Fantasy XIV trabajando en la localización al español neutro. "
        "Tu prioridad es fidelidad: el texto debe sonar natural en español SIN inventar términos ni alterar nombres propios del universo FFXIV.\n\n"
        f"{block}\n\n{base}"
    )

# ─────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────

def call_ollama(prompt: str, system: str, cfg: dict) -> str:
    url = cfg["OLLAMA_URL"].rstrip("/") + "/api/chat"
    payload = {
        "model": cfg["MODEL_NAME"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": cfg["TEMPERATURE"], 
            "top_p": 0.9, 
            "num_predict": 4096
        }
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO EN OLLAMA]: {e}")
        raise

# ─────────────────────────────────────────────
# Parseo
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Patrones de contaminación conocidos
# ─────────────────────────────────────────────

# Rutas de sistema, artefactos de terminal, prefijos que el modelo no debería incluir
_CONTAMINATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"[a-zA-Z]:[\\\/]"),                                    # rutas Windows: C:\... D:/...
    re.compile(r"\/(?:home|mnt|tmp|var|usr|opt|root|output|input|checkpoints|logs)\/"),  # rutas Unix absolutas
    re.compile(r"output\/.*\.json"),                                    # rutas de archivo JSON de salida
    re.compile(r"\[(?:ssh|tmux|bash|root|nerv)\]", re.IGNORECASE),    # artefactos de terminal SSH/tmux
    re.compile(r"(?:ssh|tmux)\b", re.IGNORECASE),                      # palabras ssh/tmux sueltas
    re.compile(r"root@\S+"),                                           # prompt de shell root@host
    re.compile(r"Translation:\s", re.IGNORECASE),                      # prefijo "Translation:" del modelo
    re.compile(r"^(?:Here(?:'s| is)|Estas son|A continuación)\b", re.IGNORECASE),  # preámbulos no pedidos
    re.compile(r"\x1b\["),                                             # secuencias de escape ANSI
    re.compile(r"^```|```$"),                                          # delimitadores de bloque markdown
]

def sanitize_response(raw: str) -> str:
    """
    Limpia la respuesta del modelo antes de parsear.
    Elimina líneas que contengan artefactos de terminal, rutas de archivo,
    secuencias de escape o cualquier basura conocida que no sea traducción.
    """
    clean_lines = []
    for line in raw.splitlines():
        is_contaminated = any(p.search(line) for p in _CONTAMINATION_PATTERNS)
        if not is_contaminated:
            clean_lines.append(line)
        # Si la línea estaba contaminada pero tenía formato N. texto,
        # la reemplazamos por una línea vacía numerada para que el parser
        # detecte el desajuste y dispare el fallback
        else:
            m = re.match(r"^\s*(\d+)\.", line)
            if m:
                clean_lines.append(f"{m.group(1)}. [CONTAMINADO]")
    return "\n".join(clean_lines)


def is_contaminated(text: str) -> bool:
    """
    Verifica si una línea ya parseada contiene contaminación residual.
    Se usa para validar cada traducción individualmente tras el parseo.
    """
    if "[CONTAMINADO]" in text:
        return True
    return any(p.search(text) for p in _CONTAMINATION_PATTERNS)


def validate_length(es: str, en: str) -> bool:
    """
    Detecta traducciones sospechosamente largas o cortas respecto al original.

    Reglas empíricas para FFXIV:
    - Una traducción al español es típicamente 10-40% más larga que el inglés.
    - Si es más de 3x el largo del original → el modelo alucinó o repitió texto.
    - Si está vacía o es menor al 20% del original → el modelo devolvió nada útil.
    - Líneas muy cortas (< 10 chars) se eximen porque son objetivos/UI de una palabra.
    """
    len_en = len(en.strip())
    len_es = len(es.strip())

    if len_en < 10:
        return True  # líneas muy cortas: no aplicar ratio (ej: "Speak with Miounne.")

    if len_es == 0:
        return False

    ratio = len_es / len_en
    return 0.2 <= ratio <= 3.0


def parse_numbered(response: str, expected: int) -> list[str] | None:
    # Primero sanitizar para eliminar basura de terminal antes de parsear
    clean = sanitize_response(response)
    lines = []
    for line in clean.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)", line)
        if m:
            text = m.group(1).strip()
            # Una línea marcada como contaminada hace fallar el batch completo
            # para que el fallback individual reintente cada entrada limpiamente
            if text == "[CONTAMINADO]":
                return None
            lines.append(text)
    return lines if len(lines) == expected else None


# ─────────────────────────────────────────────
# Post-procesado: restaurar sustantivos propios
# ─────────────────────────────────────────────

def build_restoration_map(glossary_data: dict) -> dict[str, str]:
    """
    Construye un mapa de variantes incorrectas → forma correcta para los términos
    que deben mantenerse igual en inglés (src == tgt en el glosario).

    Estrategia: detecta variantes en minúsculas para hacer la búsqueda case-insensitive,
    y restaura la forma canónica definida en el glosario.
    """
    protected_categories = {
        "characters", "factions_and_organizations",
        "key_locations_and_dungeons", "races_and_entities",
        "game_commands_and_emotes",
    }
    restoration: dict[str, str] = {}
    for cat, mapping in glossary_data.items():
        if cat not in protected_categories:
            continue
        for src, tgt in mapping.items():
            # Registrar la forma canónica para búsqueda posterior
            restoration[src.lower()] = tgt
    return restoration


def restore_proper_nouns(text: str, restoration_map: dict[str, str]) -> str:
    """
    Recorre el texto y reemplaza sustantivos propios alterados por su forma canónica.
    Opera con regex de palabras para evitar reemplazos parciales dentro de otras palabras.

    Solo restaura términos que el modelo haya modificado en capitalización o grafía menor;
    no puede detectar traducciones completamente inventadas (eso se maneja en el prompt).
    """
    if not restoration_map:
        return text

    # Ordenar por longitud descendente para priorizar matches más largos (ej: "G'raha Tia" antes que "G'raha")
    for canonical_lower, canonical in sorted(restoration_map.items(), key=lambda x: -len(x[0])):
        # Buscar el término de forma case-insensitive en el texto
        pattern = re.compile(re.escape(canonical_lower), re.IGNORECASE)
        # Solo reemplazar si la versión encontrada difiere de la canónica
        def replacer(m: re.Match) -> str:
            found = m.group(0)
            return canonical if found != canonical else found
        text = pattern.sub(replacer, text)
    
    return text

# ─────────────────────────────────────────────
# Batch con fallback
# ─────────────────────────────────────────────

def translate_batch(entries, file_type, glossary_data, cfg, log_path, batch_idx) -> dict:
    
    # Preparamos el system prompt dinámico para todo este batch
    text_lines = [t for _, t in entries]
    relevant_glossary = get_relevant_glossary(text_lines, glossary_data)
    proper_noun_fence = build_proper_noun_fence(glossary_data)
    system_prompt = build_system_prompt(file_type, relevant_glossary, proper_noun_fence)
    
    def try_batch(items):
        item_texts = [t for _, t in items]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(item_texts))
        prompt = (
            f"Traduce las siguientes {len(items)} líneas de FFXIV al español neutro.\n"
            f"IMPORTANTE: Mantén todos los nombres propios de personajes, lugares y facciones exactamente como aparecen.\n"
            f"Responde ÚNICAMENTE en formato: N. [traducción]\n"
            f"Una línea por número. Sin explicaciones ni texto adicional.\n\n{numbered}"
        )
        try:
            raw = call_ollama(prompt, system_prompt, cfg)
            parsed = parse_numbered(raw, len(items))
            if parsed is None:
                print(f"\n[ADVERTENCIA] Falló el parseo. El modelo respondió:\n{raw}\n")
            return parsed
        except Exception as e:
            print(f"\n[ERROR EN BATCH]: {e}")
            return None

    def single(key, text):
        # El fallback también usa el mismo system_prompt con todas las reglas
        try:
            raw = call_ollama(
                f"Traduce al español neutro esta línea de FFXIV. "
                f"Mantén los nombres propios sin cambiar. Devuelve solo la traducción:\n{text}",
                system_prompt, cfg
            )
            # Sanitizar también la respuesta del modo individual
            clean = sanitize_response(raw).strip()
            return clean if clean and not is_contaminated(clean) else text
        except Exception as e:
            print(f"\n[ERROR EN SINGLE]: {e}")
            return text

    # Mapa de restauración de sustantivos propios para post-procesado
    restoration_map = build_restoration_map(glossary_data)

    def postprocess(es: str) -> str:
        """Aplica correcciones post-traducción: restaura sustantivos propios alterados."""
        return restore_proper_nouns(es, restoration_map)

    def validate_entry(key: str, en: str, es: str) -> tuple[str, str | None]:
        """
        Valida una entrada traducida contra tres criterios:
        1. Contaminación: rutas, artefactos de terminal, prefijos no pedidos.
        2. Longitud: ratio ES/EN fuera del rango esperado (0.2x – 3.0x).
        3. Identidad: la traducción es igual al original en inglés.

        Devuelve (reason, es_corregido_o_None).
        - Si reason es None → entrada válida, usar es tal cual.
        - Si reason no es None → se logueará; es puede ser None si hay que reintentar.
        """
        if is_contaminated(es):
            return "contaminated", None
        if not validate_length(es, en):
            return "length_anomaly", None
        if es.strip().lower() == en.strip().lower():
            return "identical_to_source", None
        return None, es  # sin problemas

    result = {}
    translated = try_batch(entries)
    if translated is None:
        print("  Reintentando batch en 1 segundo...")
        time.sleep(1)
        translated = try_batch(entries)
    if translated is None and len(entries) > 1:
        print("  Dividiendo el batch a la mitad...")
        mid = len(entries) // 2
        a, b = try_batch(entries[:mid]), try_batch(entries[mid:])
        translated = (a + b) if (a and b) else None
    if translated is None:
        _log(log_path, batch_idx, entries, "batch_failed_fallback_individual")
        for key, text in entries:
            result[key] = postprocess(single(key, text))
        return result

    for (key, en), es in zip(entries, translated):
        es = postprocess(es)
        reason, valid_es = validate_entry(key, en, es)

        if reason is not None:
            # La entrada falló la validación: reintentar en modo individual
            if reason != "identical_to_source":
                print(f"  [!] {reason}: '{key}' → reintentando...")
            retry = postprocess(single(key, en))
            retry_reason, retry_valid = validate_entry(key, en, retry)

            if retry_reason is None:
                # El reintento es válido
                result[key] = retry_valid
                _log(log_path, batch_idx, [(key, en)], f"{reason}_recovered", retry_valid)
            else:
                # El reintento también falló: guardar el original en inglés como fallback seguro
                # y loggear para revisión manual
                print(f"  [!!] Reintento también fallido ({retry_reason}): '{key}' → guardando original EN")
                result[key] = en
                _log(log_path, batch_idx, [(key, en)], f"{reason}_unrecovered", retry)
        else:
            result[key] = valid_es

    return result

def _log(log_path, batch_idx, entries, reason, output=""):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for key, text in entries:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(), "batch": batch_idx,
                "key": key, "en": text, "es": output, "reason": reason
            }, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────

def get_ckpt_path(input_path: Path) -> Path:
    base = Path(__file__).parent
    try:
        rel = input_path.relative_to(base / "input")
        name = str(rel).replace("/", "_").replace("\\", "_")
    except ValueError:
        name = input_path.name
    return base / "checkpoints" / f"{name}.ckpt.json"

def load_checkpoint(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_checkpoint(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ─────────────────────────────────────────────
# Progreso de un archivo (para el menú)
# ─────────────────────────────────────────────

def get_file_progress(input_path: Path) -> tuple[int, int]:
    base = Path(__file__).parent
    try:
        rel = input_path.relative_to(base / "input")
    except ValueError:
        rel = Path(input_path.name)

    output_path = base / "output" / rel.parent / rel.name.replace("_en.json", "_es.json")
    ckpt = get_ckpt_path(input_path)

    try:
        total = len(json.loads(input_path.read_text(encoding="utf-8")))
    except Exception:
        return 0, 0

    if ckpt.is_file():
        try:
            done = len(json.loads(ckpt.read_text(encoding="utf-8")))
            return done, total
        except Exception:
            pass

    if output_path.is_file():
        try:
            done = len(json.loads(output_path.read_text(encoding="utf-8")))
            return done, total
        except Exception:
            pass

    return 0, total

# ─────────────────────────────────────────────
# Traducción de un archivo
# ─────────────────────────────────────────────

def translate_file(input_path: Path, cfg: dict, glossary_data: dict):
    base = Path(__file__).parent
    try:
        rel = input_path.relative_to(base / "input")
    except ValueError:
        rel = Path(input_path.name)

    output_path = base / "output" / rel.parent / rel.name.replace("_en.json", "_es.json")
    log_path    = base / "logs"   / rel.parent / (rel.stem + "_anomalies.jsonl")
    ckpt        = get_ckpt_path(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    en_data = json.loads(input_path.read_text(encoding="utf-8"))
    total   = len(en_data)

    translated  = load_checkpoint(ckpt)
    done_before = len(translated)
    if done_before:
        print(f"  Reanudando desde {done_before:,}/{total:,}...")

    def write_partial_output():
        ordered = {k: translated.get(k, en_data[k]) for k in en_data}
        output_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if done_before:
        write_partial_output()

    forced_type = detect_file_type(input_path.name)
    pending     = [(k, v) for k, v in en_data.items()
                   if k not in translated and str(v).strip()]

    if not pending:
        print(f"  Ya completo ({total:,} entradas).")
        write_partial_output()
        return

    print(f"  Pendientes: {len(pending):,} | Batches de {cfg['BATCH_SIZE']}")

    batch_size       = cfg["BATCH_SIZE"]
    checkpoint_every = cfg["CHECKPOINT_EVERY"]
    batch_count      = 0
    done_now         = 0

    def render_bar(done_now, last_batch=None):
        total_done = done_before + done_now
        pct   = (total_done / total) * 100
        bar_f = int(pct / 2)
        bar   = "#" * bar_f + "." * (50 - bar_f)
        print(f"  [{bar}] {pct:5.1f}%  {total_done:,}/{total:,}")
        if last_batch:
            for key, en_text, es_text in last_batch:
                tipo = detect_entry_type(key)
                tag  = {"journal": "JRN", "duty_list": "TODO", "dialogues": "DLG"}.get(tipo, "???")
                print(f"  [{tag}] {en_text[:60]}")
                print(f"       {es_text[:60]}")
            print()

    def run_batch(batch, file_type):
        nonlocal batch_count, done_now
        batch_count += 1
        result = translate_batch(batch, file_type, glossary_data, cfg, log_path, batch_count)
        translated.update(result)
        done_now += len(batch)
        preview = [(k, en_data[k], result[k]) for k, _ in batch[:3] if k in result]
        render_bar(done_now, preview)
        write_partial_output()
        if batch_count % checkpoint_every == 0:
            save_checkpoint(ckpt, translated)

    if forced_type:
        batches = [pending[i:i+batch_size] for i in range(0, len(pending), batch_size)]
        for batch in batches:
            run_batch(batch, forced_type)
    else:
        groups: list[tuple[str, list]] = []
        for key, text in pending:
            etype = detect_entry_type(key)
            if groups and groups[-1][0] == etype:
                groups[-1][1].append((key, text))
            else:
                groups.append((etype, [(key, text)]))

        for etype, entries in groups:
            batches = [entries[i:i+batch_size] for i in range(0, len(entries), batch_size)]
            for batch in batches:
                run_batch(batch, etype)

    write_partial_output()

    if ckpt.is_file():
        ckpt.unlink()

    print(f"\n  ✓ {output_path.relative_to(base)}")

# ─────────────────────────────────────────────
# Menú interactivo
# ─────────────────────────────────────────────

def build_menu_items(input_root: Path) -> list[dict]:
    items = []
    for folder in sorted(input_root.rglob("*")):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*_en.json"))
        if not files:
            continue

        cat_done, cat_total = 0, 0
        for f in files:
            d, t = get_file_progress(f)
            cat_done  += d
            cat_total += t

        try:
            rel = folder.relative_to(input_root)
        except ValueError:
            rel = folder

        items.append({
            "type":    "category",
            "label":   str(rel),
            "path":    folder,
            "files":   files,
            "done":    cat_done,
            "total":   cat_total,
        })

        for f in files:
            done, total = get_file_progress(f)
            try:
                frel = f.relative_to(input_root)
            except ValueError:
                frel = f
            items.append({
                "type":  "file",
                "label": f"  {frel}",
                "path":  f,
                "done":  done,
                "total": total,
            })

    return items

def show_menu(cfg: dict, input_root: Path, glossary_data: dict):
    try:
        import questionary
        from questionary import Choice
    except ImportError:
        print("\n  Instalando questionary...")
        os.system("pip install questionary -q")
        import questionary
        from questionary import Choice

    while True:
        items = build_menu_items(input_root)

        all_files  = list(input_root.rglob("*_en.json"))
        total_done = sum(get_file_progress(f)[0] for f in all_files)
        total_keys = sum(get_file_progress(f)[1] for f in all_files)
        pct_global = (total_done / total_keys * 100) if total_keys else 0

        print("\n" + "=" * 65)
        print(f"  AstralES Translator  |  {cfg['MODEL_NAME']}")
        print(f"  Progreso global: {total_done:,}/{total_keys:,}  ({pct_global:.1f}%)")
        print("=" * 65)

        choices = []
        choices.append(Choice(
            title=f"  ★ Todo el catálogo        [{total_done:,}/{total_keys:,}]",
            value="__ALL__"
        ))
        choices.append(Choice(title="  " + "─" * 55, value="__SEP__", disabled=" "))

        for item in items:
            done, total = item["done"], item["total"]
            pct   = (done / total * 100) if total else 0
            bar_f = int(pct / 10)
            bar   = "█" * bar_f + "░" * (10 - bar_f)
            status = f"[{bar}] {pct:5.1f}%  {done:,}/{total:,}"

            if item["type"] == "category":
                title = f"  ▶ {item['label']:<28} {status}"
            else:
                fname = Path(item["label"].strip()).name
                title = f"    {fname:<30} {status}"

            choices.append(Choice(title=title, value=item))

        choices.append(Choice(title="  " + "─" * 55, value="__SEP__", disabled=" "))
        choices.append(Choice(title="  Salir", value="__QUIT__"))

        selected = questionary.select(
            "",
            choices=choices,
            use_shortcuts=False,
            style=questionary.Style([
                ("selected",       "fg:#00d7af bold"),
                ("pointer",        "fg:#00d7af bold"),
                ("highlighted",    "fg:#ffffff"),
                ("answer",         "fg:#00d7af bold"),
            ])
        ).ask()

        if selected is None or selected == "__QUIT__":
            print("\n  Hasta pronto.\n")
            break

        if selected == "__SEP__":
            continue

        if selected == "__ALL__":
            files_to_run = sorted(input_root.rglob("*_en.json"))
            label = "todo el catálogo"
        elif selected["type"] == "category":
            files_to_run = selected["files"]
            label = selected["label"]
        else:
            files_to_run = [selected["path"]]
            label = selected["label"].strip()

        pending_files = []
        for f in files_to_run:
            done, total = get_file_progress(f)
            if done < total:
                pending_files.append(f)

        if not pending_files:
            print(f"\n  '{label}' ya está al 100%. Nada que traducir.")
            input("  [Enter para continuar]")
            continue

        print(f"\n  Traduciendo: {label}")
        print(f"  Archivos pendientes: {len(pending_files)}")
        print()

        for i, fpath in enumerate(pending_files, 1):
            try:
                rel = fpath.relative_to(input_root)
            except ValueError:
                rel = fpath.name
            print(f"  [{i}/{len(pending_files)}] {rel}")
            try:
                translate_file(fpath, cfg, glossary_data)
            except KeyboardInterrupt:
                print("\n\n  Interrumpido. Checkpoint guardado.")
                input("  [Enter para volver al menú]")
                break
            except Exception as e:
                print(f"\n  [Error] {e}")
            print()

        input("  [Enter para volver al menú]")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Traductor FFXIV → ES")
    parser.add_argument("--model", default=None, help="Override del modelo")
    parser.add_argument("--no-menu", action="store_true",
                        help="Procesar todo sin menú (modo Vast.ai / headless)")
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["MODEL_NAME"] = args.model

    glossary_path = Path(__file__).parent / "glossary.json"
    glossary_data = load_glossary(glossary_path)
    input_root    = Path(__file__).parent / "input"

    print(f"\n  Modelo : {cfg['MODEL_NAME']}")
    print(f"  Ollama : {cfg['OLLAMA_URL']}")

    if args.no_menu:
        files = sorted(input_root.rglob("*_en.json"))
        print(f"  Modo headless — {len(files)} archivos\n")
        for i, fpath in enumerate(files, 1):
            try:
                rel = fpath.relative_to(input_root)
            except ValueError:
                rel = fpath.name
            print(f"  [{i}/{len(files)}] {rel}")
            try:
                translate_file(fpath, cfg, glossary_data)
            except KeyboardInterrupt:
                print("\n  Interrumpido.")
                return
            except Exception as e:
                print(f"\n  [Error] {e}")
            print()
        print("  Traducción finalizada.")
    else:
        show_menu(cfg, input_root, glossary_data)

if __name__ == "__main__":
    main()