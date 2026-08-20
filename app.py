import json
import math
import re
import requests
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Simulador PDF & IA Gemini", page_icon="📂", layout="centered")

st.title("📂 Simulador PDF & IA Gemini")
st.caption("Sistema optimizado con caché de vectores y blindaje anti-repetición.")

# Inicialización del Estado de Sesión
if "banco" not in st.session_state:
    st.session_state.banco = []
if "page_texts" not in st.session_state:
    st.session_state.page_texts = {}
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

# --- FUNCIONES AUXILIARES ---

def extraer_paginas_de_texto(texto):
    if not texto:
        return []
    matches = re.findall(r'(?:páginas?|pág\.?|pp\.?|art\.?|artículo)?\s*([\d\s,y\-a]+)', texto, re.IGNORECASE)
    paginas = []
    for match in matches:
        digits = re.findall(r'\b\d+\b', match)
        paginas.extend([int(d) for d in digits if 0 < int(d) <= 1000])
    return sorted(list(set(paginas)))

def limpiar_json_markdown(texto_raw):
    """Limpia etiquetas ```json y marcas markdown de forma segura."""
    if not texto_raw:
        return ""
    cleaned = re.sub(r'^```(?:json)?', '', texto_raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'```$', '', cleaned.strip())
    return cleaned.strip()

def obtener_modelos_disponibles(api_key):
    url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){api_key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            models = data.get("models", [])
            disponibles = []
            for m in models:
                methods = m.get("supportedGenerationMethods", [])
                name = m.get("name", "").replace("models/", "")
                if "generateContent" in methods:
                    if not any(x in name for x in ["tts", "embedding", "imagen", "aqa", "bison"]):
                        disponibles.append(name)
            disponibles.sort(key=lambda x: 0 if "flash" in x else 1)
            if disponibles:
                return disponibles
    except Exception:
        pass
    return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

def obtener_embedding(texto, api_key):
    if not texto or not api_key:
        return None
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=](https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=){api_key}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": texto}]}
    }
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
        if res.status_code == 200:
            return res.json()["embedding"]["values"]
    except Exception:
        pass
    return None

def similitud_coseno(v1, v2):
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def query_gemini(prompt_text, api_key, model_list):
    for model_name in model_list:
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return {"text": parts[0]["text"], "model": model_name}
        except Exception:
            continue
    return None

def normalizar_pregunta(q):
    if not isinstance(q, dict):
        return None

    statement = q.get("statement") or q.get("pregunta") or q.get("enunciado") or "Sin enunciado"
    raw_opts = q.get("options") or q.get("opciones") or q.get("choices") or []
    
    opts = []
    for idx, o in enumerate(raw_opts):
        txt = o.get("text") if isinstance(o, dict) else str(o)
        opts.append({"index": str(idx + 1), "text": txt})
        
    correct_val = q.get("correctAnswer") or q.get("correcta") or q.get("correctIndex") or "1"
    correct_str = str(correct_val)
    
    # Manejo seguro de índices base 0 o texto
    if not correct_str.isdigit() or int(correct_str) < 1:
        match_found = False
        for o in opts:
            if o["text"].strip().lower() == correct_str.strip().lower():
                correct_str = o["index"]
                match_found = True
                break
        if not match_found:
            correct_str = "1"

    explanation = q.get("explanation") or q.get("explicacion") or ""
    page_numbers = q.get("pageNumbers") or q.get("paginas") or []
    if not page_numbers:
        page_numbers = extraer_paginas_de_texto(explanation)

    return {
        "statement": statement,
        "options": opts,
        "correctIndex": correct_str,
        "explanation": explanation,
        "pageNumbers": page_numbers,
        "embedding": q.get("embedding")  # Preserva el vector si ya existe
    }

# --- PANEL LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    st.subheader("📁 Carga de Archivos")
    json_file = st.file_uploader("Banco de Preguntas (.json)", type=["json"])
    pdf_file = st.file_uploader("Manual / Libro (.pdf)", type=["pdf"])

    if json_file:
        try:
            data = json.load(json_file)
            raw_list = data if isinstance(data, list) else data.get("questions", [])
            st.session_state.banco = [normalizar_pregunta(q) for q in raw_list if q]
            st.session_state.banco = [q for q in st.session_state.banco if q is not None]
            st.success(f"Cargadas {len(st.session_state.banco)} preguntas analizadas.")
        except Exception as e:
            st.error(f"Error en JSON: {e}")

    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
            st.session_state.total_pages = len(reader.pages)
            st.session_state.page_texts = {}
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                st.session_state.page_texts[i + 1] = txt if txt else ""
            st.success(f"PDF cargado ({st.session_state.total_pages} págs).")
        except Exception as e:
            st.error(f"Error al leer PDF: {e}")

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["✨ Generar Examen con IA", "📝 Simulador de Examen", "💾 Exportar Banco"])

with tab1:
    st.subheader("✨ Generar Examen con IA")
    
    instrucciones = st.text_area("Instrucciones extra para la IA (Opcional):", placeholder="Ej: Enfocarse en artículos específicos, plazos, montos o temas particulares...")
    
    col1, col2, col3, col4 = st.columns(4)
    p_start = col1.number_input("Pág. Inicio", min_value=1, max_value=max(1, st.session_state.total_pages), value=1)
    p_end = col2.number_input("Pág. Fin", min_value=1, max_value=max(1, st.session_state.total_pages), value=min(10, max(1, st.session_state.total_pages)))
    dificultad = col3.selectbox("Dificultad", ["básico", "intermedio", "avanzado"], index=1)
    target_count = col4.number_input("Cant. Preguntas", min_value=1, max_value=50, value=5)
    
    umbral = st.slider("🎯 Umbral de Filtro Semántico:", 0.70, 0.95, 0.83, 0.01, help="0.83 es el valor ideal: descarta la misma pregunta pero permite preguntar sobre otros detalles del mismo tema.")

    if st.button("✨ Generar Examen con IA", type="primary"):
        if not api_key:
            st.error("Ingresá tu API Key de Gemini en el menú lateral.")
        elif not st.session_state.page_texts:
            st.error("Subí un archivo PDF primero en el menú lateral.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_box = st.empty()
            logs = []

            def add_log(msg):
                logs.append(msg)
                log_box.code("\n".join(logs))

            status_text.text("🚀 Consultando modelos disponibles de Gemini...")
            model_list = obtener_modelos_disponibles(api_key)
            
            start = int(p_start)
            end = min(int(p_end), st.session_state.total_pages)
            total_pages_range = (end - start + 1)
            
            add_log(f"📄 Evaluando {total_pages_range} páginas del PDF (Págs {start} a {end})...")

            existing_bank = st.session_state.banco
            avoid_prompt = ""
            if len(existing_bank) > 0:
                enunciados = "\n".join([f"- {q.get('statement') or q.get('pregunta')}" for q in existing_bank])
                avoid_prompt = f"""
📌 REGLA DE NO REPETICIÓN CONCEPTUAL:
El usuario ya tiene en su banco las preguntas listadas abajo.
INSTRUCCIÓN ESPECÍFICA:
- NO vuelvas a hacer la misma pregunta ni a evaluar exactamente el mismo dato/concepto.
- SÍ PUEDES generar preguntas sobre el MISMO tema, artículo o capítulo, SIEMPRE Y CUANDO preguntes sobre UN DETALLE DIFERENTE (otro inciso, otro plazo, otra excepción o diferente condición) que no esté en la lista.

LISTA DE PREGUNTAS YA EXISTENTES EN EL BANCO:
{enunciados}
"""

            format_rules = """
REGLA DE LIBERTAD TOTAL DE FORMATO Y OPCIONES:
- Evaluá el material con total criterio pedagógico.
- Decidí LIBREMENTE si cada pregunta será de "Verdadero/Falso" o de "Opción Múltiple".
- Para Opción Múltiple: NO hay límites en la cantidad de opciones.
- Para Verdadero/Falso: Usá exactamente 2 alternativas: ["Verdadero", "Falso"].
- ES OBLIGATORIO incluir el número exacto de página del PDF en la "explanation" (ej: "Página 12") y en el arreglo "pageNumbers": [12].
- Asegurate de que "correctAnswer" sea la copia idéntica del texto de una de las opciones listadas.
"""

            raw_questions = []

            if total_pages_range > 35:
                add_log("⚡ Modo PDF Extenso Activado: Procesando por sub-lotes.")
                batch_count = min(3, math.ceil(total_pages_range / 35))
                questions_per_batch = math.ceil(target_count / batch_count)
                batch_page_size = total_pages_range / batch_count

                for b in range(batch_count):
                    b_start = math.floor(start + (b * batch_page_size))
                    b_end = end if (b == batch_count - 1) else math.floor(start + ((b + 1) * batch_page_size) - 1)
                    count_for_batch = (target_count - len(raw_questions)) if (b == batch_count - 1) else questions_per_batch
                    if count_for_batch <= 0:
                        break

                    batch_text = ""
                    for p in range(b_start, b_end + 1):
                        batch_text += f"[Página {p}]: {st.session_state.page_texts.get(p, '')}\n"

                    add_log(f"▶ [Sub-lote {b + 1}/{batch_count}] Páginas {b_start} a {b_end}...")

                    batch_prompt = f"""Actúa como docente evaluador universitario experto.
Analiza las páginas {b_start} a {b_end} del documento.
Genera EXACTAMENTE {count_for_batch} PREGUNTAS evaluando el contenido sustancial.
Nivel: {dificultad.upper()}.
{format_rules}
{avoid_prompt}
{f'INSTRUCCIÓN EXTRA: {instrucciones}' if instrucciones else ''}

Devuelve un JSON estricto:
[
  {{
    "statement": "Enunciado claro o afirmación a evaluar",
    "options": ["Opción 1", "Opción 2", "..."],
    "correctAnswer": "Opción 1",
    "explanation": "Fundamentación técnica e indicación de página (ej. Según la Página {b_start}...)",
    "pageNumbers": [{b_start}]
  }}
]

CONTENIDO:
{batch_text}"""

                    res = query_gemini(batch_prompt, api_key, model_list)
                    if res and res.get("text"):
                        cleaned = limpiar_json_markdown(res["text"])
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, list):
                            raw_questions.extend(parsed)
                        add_log(f"🟢 Sub-lote {b + 1} recibido desde {res['model']}.")

                    pct = int(((b + 1) / batch_count) * 80)
                    progress_bar.progress(pct)

            else:
                add_log("⚡ Modo Conexión Única Rápida Activado por Bloques Temáticos...")
                blocks_text_prompt = ""
                chunk_size = total_pages_range / target_count

                for i in range(target_count):
                    chunk_start = math.floor(start + (i * chunk_size))
                    chunk_end = end if (i == target_count - 1) else math.floor(start + ((i + 1) * chunk_size) - 1)
                    if chunk_end < chunk_start:
                        chunk_end = chunk_start

                    chunk_text = ""
                    for p in range(chunk_start, chunk_end + 1):
                        chunk_text += f"[Página {p}]: {st.session_state.page_texts.get(p, '')}\n"
                    blocks_text_prompt += f"\n=== BLOQUE TEMÁTICO {i + 1} (Páginas {chunk_start} a {chunk_end}) ===\n{chunk_text}\n"

                full_prompt = f"""Actúa como docente evaluador universitario experto.
Analiza el documento dividido en {target_count} bloques temáticos.
Genera EXACTAMENTE {target_count} PREGUNTAS (1 por cada bloque).
Nivel: {dificultad.upper()}.
{format_rules}
{avoid_prompt}
{f'INSTRUCCIÓN ADICIONAL: {instrucciones}' if instrucciones else ''}

Devuelve un JSON estricto:
[
  {{
    "statement": "Enunciado o afirmación a evaluar",
    "options": ["Opción A", "Opción B", "..."],
    "correctAnswer": "Opción A",
    "explanation": "Fundamentación técnica indicando la página correspondiente (ej. Página {start})",
    "pageNumbers": [{start}]
  }}
]

CONTENIDO POR BLOQUES:
{blocks_text_prompt}"""

                progress_bar.progress(40)
                res = query_gemini(full_prompt, api_key, model_list)
                if res and res.get("text"):
                    cleaned = limpiar_json_markdown(res["text"])
                    raw_questions = json.loads(cleaned)
                    add_log(f"🟢 Examen recibido desde {res['model']}.")

            progress_bar.progress(85)

            if raw_questions:
                add_log("3/3 Evaluando vectores semánticos con almacenamiento en caché...")
                
                # Optimización: Carga e indexación en caché de los vectores del banco actual
                for q in st.session_state.banco:
                    if not q.get("embedding"):
                        q["embedding"] = obtener_embedding(q["statement"], api_key)

                aceptadas = 0
                rechazadas = 0

                for raw_q in raw_questions:
                    norm = normalizar_pregunta(raw_q)
                    if not norm:
                        continue
                        
                    emb_nuevo = obtener_embedding(norm["statement"], api_key)
                    norm["embedding"] = emb_nuevo

                    es_duplicada = False
                    if emb_nuevo:
                        for prev_q in st.session_state.banco:
                            prev_emb = prev_q.get("embedding")
                            if prev_emb and similitud_coseno(emb_nuevo, prev_emb) >= umbral:
                                es_duplicada = True
                                break

                    if es_duplicada:
                        rechazadas += 1
                    else:
                        st.session_state.banco.append(norm)
                        aceptadas += 1

                progress_bar.progress(100)
                status_text.empty()
                add_log(f"✨ ¡Proceso terminado! Se agregaron {aceptadas} preguntas inéditas. ({rechazadas} rechazadas por ser la misma pregunta).")
                st.balloons()
                st.success(f"🎉 Se agregaron {aceptadas} preguntas inéditas al banco.")

with tab2:
    st.subheader("📝 Simulador de Examen")
    if not st.session_state.banco:
        st.info("El banco está vacío. Generá preguntas con la IA o cargá un JSON.")
    else:
        st.write(f"Total en banco: **{len(st.session_state.banco)} preguntas**")
        num_q = st.number_input("Seleccionar pregunta:", min_value=1, max_value=len(st.session_state.banco), value=1)
        idx = num_q - 1
        
        q_act = st.session_state.banco[idx]
        statement_text = q_act.get("statement") or q_act.get("pregunta") or q_act.get("enunciado") or "Sin enunciado"
        
        st.markdown(f"### **{num_q}. {statement_text}**")
        
        raw_opts = q_act.get("options") or q_act.get("opciones") or []
        opts_text = [o["text"] if isinstance(o, dict) else str(o) for o in raw_opts]
        
        if opts_text:
            eleccion = st.radio("Seleccioná tu respuesta:", opts_text, key=f"quiz_rad_{idx}")
            
            if st.button("Comprobar Respuesta", key=f"quiz_btn_{idx}"):
                corr_val = q_act.get("correctIndex") or q_act.get("correcta") or 1
                corr_idx = int(corr_val) - 1 if str(corr_val).isdigit() else 0
                
                if 0 <= corr_idx < len(opts_text):
                    if opts_text.index(eleccion) == corr_idx:
                        st.success("✅ ¡Correcto!")
                    else:
                        st.error(f"❌ Incorrecto. La opción correcta era: **{opts_text[corr_idx]}**")
                
                exp_text = q_act.get("explanation") or q_act.get("explicacion") or "Sin explicación provista."
                st.info(f"💡 **Explicación:** {exp_text}")
                
                p_nums = q_act.get("pageNumbers") or extraer_paginas_de_texto(exp_text)
                if p_nums:
                    st.markdown(f"**📄 Páginas de referencia:** {', '.join(map(str, p_nums))}")
                    for p in p_nums:
                        if p in st.session_state.page_texts and st.session_state.page_texts[p]:
                            with st.expander(f"📄 Ver fragmento original de la Página {p}"):
                                st.text_area(f"Texto Página {p}:", value=st.session_state.page_texts[p], height=180, disabled=True, key=f"exp_txt_{idx}_{p}")

with tab3:
    st.subheader("💾 Exportar Banco de Preguntas")
    if st.session_state.banco:
        # Remover propiedad 'embedding' antes de descargar para no abultar el JSON
        exportable = []
        for q in st.session_state.banco:
            item = {k: v for k, v in q.items() if k != "embedding"}
            exportable.append(item)
            
        json_str = json.dumps(exportable, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar preguntas.json actualizado",
            data=json_str,
            file_name="preguntas.json",
            mime="application/json"
        )
    else:
        st.write("Sin preguntas disponibles para descargar.")
