"""Etapa 5: buscar los párrafos más parecidos a una pregunta. Dos recuperadores para compararlos:

  - TF-IDF + coseno (clase S13_2): línea base, encuentra palabras en común.
  - denso: embeddings + FAISS (clase S14_1): encuentra significado aunque las palabras sean otras.

Los dos devuelven una lista de diccionarios con la misma forma (texto, periodo, documento, pagina,
similitud...), así `evaluacion.py` puede compararlos con la misma función.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def detectar_periodos(pregunta):
    """Gobiernos que menciona la pregunta.

    Prioridad: un rango ("2014-2018") o un año ("2015") explícitos mandan sobre el nombre del presidente
    ("Santos" son dos periodos, pero "Santos 2014-2018" es solo uno).
    """
    q = pregunta.lower()
    explicitos = []
    for m in re.finditer(r"\b(20[0-2]\d)\s*(?:-|–|a)\s*(20[0-3]\d)\b", q):
        if f"{m.group(1)}-{m.group(2)}" in config.PERIODOS:
            explicitos.append(f"{m.group(1)}-{m.group(2)}")
    # Los rangos se quitan antes de mirar años sueltos; si no, "2018-2022" también sumaría el año 2022
    # y con él el periodo 2022-2026.
    q_sin_rangos = re.sub(r"\b20[0-2]\d\s*(?:-|–|a)\s*20[0-3]\d\b", " ", q)
    for anio in re.findall(r"\b(20[0-2]\d)\b", q_sin_rangos):
        for p in config.PERIODOS:
            ini, fin = map(int, p.split("-"))
            if ini <= int(anio) < fin or (int(anio) == fin and p == config.PERIODOS[-1]):
                explicitos.append(p)
    if explicitos:
        return sorted(set(explicitos))
    por_nombre = []
    for nombre, periodos in config.PRESIDENTES.items():
        if re.search(rf"\b{nombre}\b", q):
            por_nombre += periodos
    return sorted(set(por_nombre))


# ------------------------------------------------------------------ denso (S14_1)
def buscar_denso(vs, pregunta, k=5, periodos=None):
    """Búsqueda en el índice FAISS. Si hay `periodos`, solo devuelve párrafos de esos gobiernos."""
    if periodos:
        filtro = lambda meta: meta["periodo"] in periodos          # noqa: E731
        # LangChain filtra DESPUÉS de traer `fetch_k` vecinos. Con el valor por defecto (20), un gobierno poco
        # relacionado con la pregunta se queda con menos de k resultados (o con ninguno). Se traen todos.
        fetch_k = vs.index.ntotal
    else:
        filtro, fetch_k = None, max(k, 20)
    pares = vs.similarity_search_with_score(pregunta, k=k, filter=filtro, fetch_k=fetch_k)
    resultados = []
    for doc, distancia in pares:
        m = doc.metadata
        resultados.append({
            "texto": doc.page_content, "periodo": m["periodo"], "presidente": m["presidente"],
            "documento": m["documento"], "pagina": m["pagina"], "tema": m.get("tema", ""),
            # Con vectores de norma 1, FAISS devuelve la distancia al cuadrado = 2 - 2*coseno
            "similitud": float(1 - distancia / 2),
        })
    return resultados


# ------------------------------------------------------------------ TF-IDF (S13_2)
def crear_tfidf(df):
    """Ajusta TF-IDF sobre todos los párrafos."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matriz = vec.fit_transform(df["texto"])
    return {"vec": vec, "matriz": matriz, "df": df.reset_index(drop=True)}


def buscar_tfidf(modelo, pregunta, k=5, periodos=None):
    """Los k párrafos con mayor coseno TF-IDF respecto a la pregunta (filtrando por gobierno antes de ordenar)."""
    df = modelo["df"]
    sim = cosine_similarity(modelo["vec"].transform([pregunta]), modelo["matriz"]).ravel()
    if periodos:
        sim = np.where(df["periodo"].isin(periodos), sim, -1.0)
    resultados = []
    for i in np.argsort(sim)[::-1][:k]:
        if sim[i] <= 0:
            break                                   # sin ninguna palabra en común: no se inventa un resultado
        f = df.iloc[i]
        resultados.append({"texto": f["texto"], "periodo": f["periodo"], "presidente": f["presidente"],
                           "documento": f["documento"], "pagina": int(f["pagina"]),
                           "tema": f.get("tema", ""), "similitud": float(sim[i])})
    return resultados
