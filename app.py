import json
import math
import requests
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Motor de Estudio IA", page_icon="🧠", layout="centered")

st.title("🧠 Motor de Estudio Semántico")
st.caption("Generación continua de preguntas de examen sin repetición mediante Embeddings.")

# Inicialización de estado
if "banco" not in st.session_state:
    st.session_state.banco = []
if "pdf_texto" not in st.session_state:
    st.session_state.pdf_texto = ""

# Funciones de comunicación con Gemini
def obtener_embedding(texto, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
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
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def generar_con_gemini(prompt, api_key):
    # Probar lista de modelos compatibles en orden
    modelos = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    
    for modelo in modelos:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.warning(f"⚠️ El modelo {modelo} devolvió error HTTP {res.status_code}: {res.text}")
        except Exception as e:
            st.warning(f"⚠️ Error al conectar con {modelo}: {e}")
            
    st.error("❌ No se pudo obtener respuesta de ningún modelo de Gemini. Verificá tu API Key.")
    return None

# Panel Lateral
with st.sidebar:
    st.header("⚙️ Ajustes y Archivos")
    api_key = st.text_input("Gemini API Key:", type="password", help="Tu clave de Google AI Studio")
    
    json_file = st.file_uploader("Cargar JSON Existente (.json)", type=["json"])
    pdf_file = st.file_uploader("Cargar PDF de Estudio (.pdf)", type=["pdf"])

    if json_file:
        try:
            data = json.load(json_file)
            st.session_state.banco = data if isinstance(data, list) else data.get("questions", [])
            st.success(f"Cargadas {len(st.session_state.banco)} preguntas.")
        except Exception as e:
            st.error(f"Error al leer JSON: {e}")

    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
            texto = ""
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    texto += f"\n[Página {i+1}]: " + txt
            st.session_state.pdf_texto = texto
            st.success(f"PDF procesado ({len(reader.pages)} págs).")
        except Exception as e:
            st.error(f"Error al leer PDF: {e}")

# Pestañas Principales
tab1, tab2, tab3 = st.tabs(["✨ Generador Antiduplicados", "📝 Simulador de Examen", "💾 Exportar Banco"])

with tab1:
    st.subheader("Generar Preguntas Inéditas con IA")
    col1, col2 = st.columns(2)
    cant = col1.number_input("Cantidad a generar:", min_value=1, max_value=20, value=5)
    umbral = col2.slider("Umbral de similitud (Filtro):", 0.60, 0.95, 0.80, 0.05)
    instrucciones = st.text_area("Instrucciones o temas específicos:", placeholder="Ej: Centrarse en plazos, sanciones o artículos del capítulo 2.")

    if st.button("🚀 Generar Preguntas", type="primary"):
        if not api_key:
            st.error("Ingresá tu API Key de Gemini en el menú lateral.")
        elif not st.session_state.pdf_texto:
            st.error("Subí un archivo PDF en el menú lateral.")
        else:
            with st.spinner("1/3 Procesando vectores semánticos del banco actual..."):
                vectores_existentes = []
                for q in st.session_state.banco:
                    txt = q.get("pregunta") or q.get("statement") or ""
                    if txt:
                        emb = obtener_embedding(txt, api_key)
                        if emb:
                            vectores_existentes.append(emb)

            with st.spinner("2/3 Generando preguntas con la API de Gemini..."):
                prompt = f"""
                Actúa como un profesor universitario riguroso.
                Analiza el siguiente texto y genera EXACTAMENTE {cant} preguntas de evaluación sobre detalles clave.
                {f'INSTRUCCIÓN ADICIONAL: {instrucciones}' if instrucciones else ''}

                Responde UNICAMENTE en formato JSON estricto con esta estructura:
                [
                  {{
                    "pregunta": "Enunciado claro y concreto",
                    "opciones": ["Opción A", "Opción B", "Opción C", "Opción D"],
                    "correcta": 1,
                    "explicacion": "Breve explicación técnica de la respuesta"
                  }}
                ]

                TEXTO DE ESTUDIO:
                {st.session_state.pdf_texto[:30000]}
                """
                
                raw_json = generar_con_gemini(prompt, api_key)

            if raw_json:
                with st.spinner("3/3 Aplicando filtro semántico antiduplicados..."):
                    try:
                        cleaned = raw_json.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                        
                        nuevas_preguntas = json.loads(cleaned)
                        aceptadas = 0
                        rechazadas = 0

                        for q in nuevas_preguntas:
                            txt_nuevo = q.get("pregunta", "")
                            emb_nuevo = obtener_embedding(txt_nuevo, api_key)
                            
                            es_duplicada = False
                            if emb_nuevo and vectores_existentes:
                                for prev_emb in vectores_existentes:
                                    if similitud_coseno(emb_nuevo, prev_emb) >= umbral:
                                        es_duplicada = True
                                        break
                            
                            if es_duplicada:
                                rechazadas += 1
                            else:
                                st.session_state.banco.append(q)
                                if emb_nuevo:
                                    vectores_existentes.append(emb_nuevo)
                                aceptadas += 1

                        st.success(f"🎉 ¡Proceso finalizado! Se agregaron {aceptadas} preguntas únicas. ({rechazadas} rechazadas por similitud conceptual).")
                    except Exception as e:
                        st.error(f"Error al interpretar la respuesta JSON de la IA: {e}")

with tab2:
    st.subheader("Simulador de Examen")
    if not st.session_state.banco:
        st.info("El banco está vacío. Subí un JSON o generá preguntas nuevas.")
    else:
        st.write(f"Total en banco: **{len(st.session_state.banco)} preguntas**")
        num_q = st.number_input("Pregunta número:", min_value=1, max_value=len(st.session_state.banco), value=1)
        idx = num_q - 1
        
        q_act = st.session_state.banco[idx]
        enunciado = q_act.get("pregunta") or q_act.get("statement") or "Sin enunciado"
        st.markdown(f"### **{num_q}. {enunciado}**")
        
        raw_opts = q_act.get("opciones") or q_act.get("options") or []
        opts = [o.get("text") if isinstance(o, dict) else str(o) for o in raw_opts]
        
        if opts:
            eleccion = st.radio("Seleccioná tu respuesta:", opts, key=f"rad_{idx}")
            
            if st.button("Comprobar Respuesta", key=f"btn_{idx}"):
                corr_val = q_act.get("correcta") or q_act.get("correctIndex") or 1
                corr_idx = int(corr_val) - 1 if str(corr_val).isdigit() else 0
                
                if 0 <= corr_idx < len(opts):
                    if opts.index(eleccion) == corr_idx:
                        st.success("✅ ¡Correcto!")
                    else:
                        st.error(f"❌ Incorrecto. La opción correcta era: **{opts[corr_idx]}**")
                st.info(f"💡 **Explicación:** {q_act.get('explicacion') or q_act.get('explanation', 'Sin explicación.')}")

with tab3:
    st.subheader("Descargar Banco de Preguntas")
    if st.session_state.banco:
        json_str = json.dumps(st.session_state.banco, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar preguntas.json",
            data=json_str,
            file_name="preguntas_depuradas.json",
            mime="application/json"
        )
    else:
        st.write("Aún no hay preguntas para descargar.")
