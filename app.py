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
    cleaned = re.sub(r'
