"""Utilidades compartidas del análisis PND; no descarga ni ejecuta al importar."""
import re
import hashlib
import unicodedata
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score

OTROS = "Otros / sin tema claro"

def preparar_texto(texto):
    """Misma transformación para entrenamiento e inferencia; conserva números y negación."""
    texto = unicodedata.normalize("NFC", str(texto)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texto)).strip()

def firma_texto(texto):
    return hashlib.sha256(preparar_texto(texto).encode("utf-8")).hexdigest()

def limitar_segmentos(texto, max_caracteres=2000):
    """Límite estricto, incluso para oraciones largas y tokens sin espacios."""
    if max_caracteres < 1:
        raise ValueError("max_caracteres debe ser positivo")
    partes = []
    while len(texto) > max_caracteres:
        corte = texto.rfind(" ", 0, max_caracteres + 1)
        if corte <= 0:
            corte = max_caracteres
        partes.append(texto[:corte].strip())
        texto = texto[corte:].strip()
    if texto:
        partes.append(texto)
    return partes

def muestra_validacion(df, por_estrato=10, semilla=42):
    """Muestra por periodo y etiqueta automática, incluidos Otros; no es muestra proporcional."""
    seleccion = []
    for _, grupo in df.groupby(["periodo", "tema_principal"], sort=True):
        seleccion.append(grupo.sample(min(por_estrato, len(grupo)), random_state=semilla))
    out = pd.concat(seleccion).sort_values("id").copy()
    for col in ["tema_revisor_1", "tema_revisor_2", "tema_consenso", "observaciones"]:
        out[col] = ""
    return out

def leer_consenso(ruta, corpus):
    """Rechaza IDs duplicados, desconocidos y clases fuera de la taxonomía."""
    if not Path(ruta).exists():
        return pd.DataFrame()
    revision = pd.read_csv(ruta).fillna("")
    requeridas = {"id", "tema_consenso"}
    if not requeridas.issubset(revision):
        raise ValueError("La revisión requiere id y tema_consenso")
    revision["tema_consenso"] = revision.tema_consenso.str.strip()
    revision = revision[revision.tema_consenso.ne("")].copy()
    if revision.id.duplicated().any():
        raise ValueError("IDs duplicados en validación")
    if not set(revision.id).issubset(set(corpus.id)):
        raise ValueError("La revisión incluye IDs ajenos al corpus")
    validas = set(corpus.tema_principal) | {OTROS}
    if not set(revision.tema_consenso).issubset(validas):
        raise ValueError("Etiqueta de consenso fuera de la taxonomía")
    validar_revision(revision, corpus)
    return corpus.merge(revision[["id", "tema_consenso"]], on="id", validate="one_to_one")

def reservar_revision(df, ruta="corpus/validacion_manual.csv"):
    """Reserva TODOS los IDs muestreados, aun sin revisar, y sus duplicados textuales."""
    if not Path(ruta).exists():
        return df.copy()
    revision = pd.read_csv(ruta)
    validar_revision(revision, df)
    ids = revision["id"]
    firmas = set(df.loc[df.id.isin(ids), "texto"].map(firma_texto))
    return df.loc[~df.texto.map(firma_texto).isin(firmas)].copy()

def nuevo_modelo(min_df=3):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preparar_texto,
            lowercase=False, ngram_range=(1, 2), min_df=min_df,
            max_df=0.95, sublinear_tf=True, max_features=50000)),
        ("clasificador", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def separar_periodo(df, periodo):
    """Separa por gobierno y elimina duplicados exactos normalizados del entrenamiento."""
    prueba = df[df.periodo.eq(periodo)].copy()
    train = df[~df.periodo.eq(periodo)].copy()
    firmas = set(prueba.texto.map(firma_texto))
    train = train[~train.texto.map(firma_texto).isin(firmas)]
    return train, prueba

def acuerdo_revisores(revision):
    d = revision.dropna(subset=["tema_revisor_1", "tema_revisor_2"])
    d = d[d.tema_revisor_1.str.strip().ne("") & d.tema_revisor_2.str.strip().ne("")]
    if d.empty:
        return {"n_doble_revision": 0, "kappa": None}
    return {"n_doble_revision": len(d), "kappa": cohen_kappa_score(d.tema_revisor_1, d.tema_revisor_2)}


def validar_revision(revision, corpus):
    """Evita aplicar anotaciones obsoletas a otro corpus o segmentación."""
    if revision.id.duplicated().any() or not set(revision.id).issubset(set(corpus.id)):
        raise ValueError("Muestra manual incompatible con los IDs del corpus")
    if "texto" in revision:
        unido = revision[["id", "texto"]].merge(corpus[["id", "texto"]], on="id", suffixes=("_revision", "_actual"), validate="one_to_one")
        if not unido.texto_revision.eq(unido.texto_actual).all():
            raise ValueError("El texto cambió: archivar la revisión y regenerar la muestra")
