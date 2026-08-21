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
        "TEMPERATURE":      0.6,  # Temperatura baja para acatar reglas, pero sin romper gramática
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

def build_system_prompt(file_type: str, relevant_glossary: str) -> str:
    base = f"""GLOSARIO PARA ESTE TEXTO (aplica siempre, reemplazo exacto):
{relevant_glossary}

REGLAS ESTRICTAS DE LOCALIZACIÓN:
- Estás haciendo LOCALIZACIÓN PROFESIONAL, no traducción palabra por palabra. 
- Adapta modismos, dichos y expresiones al español neutro según el contexto.
- Mantén un vocabulario rico, natural y con tono de fantasía épica.
- Devuelve ÚNICAMENTE las líneas traducidas en el formato numerado indicado.
- Sin explicaciones, notas, comentarios ni formato extra.
- Nunca traduzcas comandos que comiencen con / (ej: /wave, /stretch).
- Nunca traduzcas nombres propios de personajes, facciones o zonas si no están en el glosario. Mantenlos en inglés.
- Jamás repitas el texto en inglés como si fuera la traducción."""

    specific = {
        "duty_list":     "TIPO: Objetivo de misión (Duty List)\n- Imperativo conciso: \"Speak with\" → \"Habla con\", \"Find\" → \"Busca a\", \"Head to\" → \"Dirígete a\".\n- Texto breve y directo, sin perder el tono de aventura.",
        "journal":       "TIPO: Diario de misión (Journal)\n- Tono narrativo, fluido y literario, como una crónica de aventuras. Tercera persona o resumen de eventos.\n- Adapta expresiones idiomáticas a equivalentes naturales en español.",
        "dialogues":     "TIPO: Diálogo de personaje\n- Preserva el tono, acento y registro de cada personaje (formal/informal, arcaico, rústico, etc.).\n- Sonido de doblaje profesional de fantasía, NUNCA traducción literal. Fluye natural en español.\n- Respeta puntuación original (puntos suspensivos, exclamaciones, tartamudeos).",
        "actions":       "TIPO: Acción de combate\n- Nombres: cortos, impactantes, con sabor a fantasía. Ej: \"Bloodbath\" → \"Baño de sangre\".\n- Descripciones: claras y precisas en términos de mecánica de juego.",
        "status_effects":"TIPO: Efecto de estado\n- Nombres concisos (1-3 palabras). HP/MP se mantienen igual.\n- Descripciones: explican claramente el efecto mecánico.",
        "traits":        "TIPO: Rasgo de clase\n- Nombres descriptivos y coherentes con la acción que mejoran.",
        "ui":            "TIPO: Interfaz de usuario (UI)\n- Texto extremadamente conciso.\n- Preserva variables: {0}, {1}, %s exactamente igual.",
    }

    block = specific.get(file_type, "TIPO: Texto general de videojuego.")
    return (
        "Eres un experto en localización de videojuegos trabajando en Final Fantasy XIV. "
        "Tu objetivo es que el texto suene como si hubiera sido escrito originalmente en español, "
        "propio de una novela de fantasía épica, evitando calcos del inglés y traducciones literales torpes.\n\n"
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

def parse_numbered(response: str, expected: int) -> list[str] | None:
    lines = []
    for line in response.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)", line)
        if m:
            lines.append(m.group(1).strip())
    return lines if len(lines) == expected else None

# ─────────────────────────────────────────────
# Batch con fallback
# ─────────────────────────────────────────────

def translate_batch(entries, file_type, glossary_data, cfg, log_path, batch_idx) -> dict:
    
    # Preparamos el system prompt dinámico para todo este batch
    text_lines = [t for _, t in entries]
    relevant_glossary = get_relevant_glossary(text_lines, glossary_data)
    system_prompt = build_system_prompt(file_type, relevant_glossary)
    
    def try_batch(items):
        item_texts = [t for _, t in items]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(item_texts))
        prompt = (
            f"Traduce las siguientes {len(items)} líneas de FFXIV al español neutro, "
            f"manteniendo el tono fantástico y épico del original.\n"
            f"Responde ÚNICAMENTE en formato: N. [traducción]\n"
            f"Una línea por número. Sin explicaciones.\n\n{numbered}"
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
        try:
            out = call_ollama(
                f"Traduce al español neutro, con tono fantástico y épico:\n{text}",
                system_prompt, cfg
            )
            return out if out.strip() else text
        except Exception as e:
            print(f"\n[ERROR EN SINGLE]: {e}")
            return text

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
            result[key] = single(key, text)
        return result

    for (key, en), es in zip(entries, translated):
        if es.strip().lower() == en.strip().lower():
            retry = single(key, en)
            if retry.strip().lower() != en.strip().lower():
                es = retry
            _log(log_path, batch_idx, [(key, en)], "identical_to_source", es)
        result[key] = es
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