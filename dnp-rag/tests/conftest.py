"""Datos sintéticos para las pruebas: no usan internet, GPU ni modelos descargados."""
import hashlib
import re

import numpy as np
import pandas as pd
import pytest
from langchain_core.embeddings import Embeddings

from dnp_rag import config, diccionario

PLANTILLAS = {
    "Educación": "El gobierno ampliará la cobertura educativa, mejorará la calidad de la educación escolar y fortalecerá a los docentes y estudiantes de colegios y universidades de todo el país durante el cuatrienio.",
    "Vivienda": "Se promoverá el acceso a vivienda de interés social con subsidios habitacionales, crédito hipotecario y mejoramiento de barrios para reducir el déficit habitacional en las ciudades del país.",
    "Empleo": "La política buscará reducir el desempleo y la informalidad, mejorar el mercado laboral y la empleabilidad de los trabajadores jóvenes mediante empleos formales y trabajo decente en las regiones.",
    "Salud": "El sistema de salud garantizará el aseguramiento, fortalecerá los hospitales y la atención a pacientes, reducirá la mortalidad y ampliará la vacunación y la nutrición infantil en el territorio.",
}
PRESIDENTES = {"2006-2010": "Álvaro Uribe Vélez", "2010-2014": "Juan Manuel Santos", "2014-2018": "Juan Manuel Santos",
               "2018-2022": "Iván Duque Márquez", "2022-2026": "Gustavo Petro"}


# palabras funcionales: un modelo real no las toma como "tema", así que el de mentira tampoco
FUNCIONALES = {"de", "la", "el", "en", "y", "los", "las", "del", "se", "que", "un", "una", "para", "con", "por", "al", "es", "o", "a", "su", "sus"}


class EmbeddingsFalsos(Embeddings):
    """Embeddings de juguete (bolsa de palabras con hash) con la misma interfaz que HuggingFaceEmbeddings."""

    def __init__(self, dim=256):
        self.dim = dim

    def _vector(self, texto):
        v = np.zeros(self.dim)
        for palabra in re.findall(r"[a-záéíóúñü]+", texto.lower()):
            if palabra in FUNCIONALES:
                continue
            v[int(hashlib.md5(palabra.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()          # norma 1, como con normalize_embeddings=True

    def embed_documents(self, textos):
        return [self._vector(t) for t in textos]

    def embed_query(self, texto):
        return self._vector(texto)


def _codigo(n):
    """Código único solo de letras (los embeddings falsos ignoran los dígitos)."""
    letras, n = "", n + 1
    while n:
        n, r = divmod(n - 1, 26)
        letras = chr(97 + r) + letras
    return "cod" + letras


@pytest.fixture()
def embeddings():
    return EmbeddingsFalsos()


@pytest.fixture()
def corpus():
    """200 párrafos: 5 gobiernos x 10 páginas x 4 temas, cada uno distinguible por un código."""
    filas, n = [], 0
    for periodo in config.PERIODOS:
        for pagina in range(1, 11):
            for base in PLANTILLAS.values():
                n += 1
                filas.append({"id": n, "periodo": periodo, "presidente": PRESIDENTES[periodo], "documento": f"doc-{periodo}",
                              "pagina": pagina, "texto": f"{base} Versión {_codigo(n)}: este párrafo es del periodo {periodo}."})
    df = pd.DataFrame(filas)
    df["n_palabras"] = df["texto"].str.split().str.len()
    return diccionario.etiquetar(df)


@pytest.fixture()
def datos_tmp(tmp_path):
    """Manda PDF, corpus e índice a una carpeta temporal y la restaura al terminar."""
    anterior = config.DATA_DIR
    config.usar_carpeta_datos(tmp_path / "datos")
    yield tmp_path / "datos"
    config.usar_carpeta_datos(anterior)
