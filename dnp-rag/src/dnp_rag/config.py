"""Rutas y parámetros del proyecto. Todo lo que se quiera cambiar va en este archivo."""
import os
from pathlib import Path

# ------------------------------------------------------------------ RUTAS
RAIZ_REPO = Path(__file__).resolve().parents[2]    # carpeta del repositorio (dnp-rag/)
GOLD_DIR = RAIZ_REPO / "data" / "gold"             # preguntas anotadas por personas (esta SÍ se sube a Git)


def usar_carpeta_datos(ruta):
    """Define dónde se guardan PDF, corpus e índice.

    En Colab conviene una carpeta de Drive para no perder todo al cerrar la sesión.
    Los demás módulos leen estas rutas con `config.PDF_DIR`, etc. (no copian el valor).
    """
    global DATA_DIR, PDF_DIR, CORPUS_DIR, INDICE_DIR, RESULTADOS_DIR
    DATA_DIR = Path(ruta)
    PDF_DIR = DATA_DIR / "pdfs"
    CORPUS_DIR = DATA_DIR / "corpus"
    INDICE_DIR = DATA_DIR / "indice"
    RESULTADOS_DIR = DATA_DIR / "resultados"
    for carpeta in (PDF_DIR, CORPUS_DIR, INDICE_DIR, RESULTADOS_DIR, GOLD_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)


usar_carpeta_datos(os.environ.get("DNP_DATA_DIR", RAIZ_REPO / "data"))

SEMILLA = 42

# ------------------------------------------------------------------ DESCARGA (de la v4)
URL_BASE = "https://www.dnp.gov.co/plan-nacional-desarrollo/Paginas/default.aspx"
URL_PND_ACTUAL = "https://www.dnp.gov.co/plan-nacional-desarrollo/Paginas/pnd-2022-2026.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# periodo -> nombres de archivo (sin .pdf) de los tomos que se descargan
SELECCION = {
    "2022-2026": ["plan-nacional-de-desarrollo-2022-2026-colombia-potencia-mundial-de-la-vida"],
    "2018-2022": ["tomo-1-dnp", "tomo-2-pnd"],
    "2014-2018": ["PND 2014-2018 Tomo 1 internet", "PND 2014-2018 Tomo 2 internet"],
    "2010-2014": ["PND2010-2014 Tomo I CD", "PND2010-2014 Tomo II CD"],
    "2006-2010": ["PND_Tomo_1", "PND_Tomo_2"],
}
PERIODOS = sorted(SELECCION)
# Para detectar de qué gobierno habla una pregunta ("el plan de Santos"). Santos tuvo dos periodos.
PRESIDENTES = {"uribe": ["2006-2010"], "santos": ["2010-2014", "2014-2018"],
               "duque": ["2018-2022"], "petro": ["2022-2026"]}

# ------------------------------------------------------------------ EXTRACCIÓN Y SEGMENTACIÓN (de la v4)
MIN_CARACTERES_PAGINA = 50    # página con menos texto que esto se cuenta como "sin texto"
MIN_PALABRAS = 20             # los bloques con menos palabras se descartan
MAX_CARACTERES = 2000         # los párrafos más largos se dividen por oraciones
UMBRAL_NUMERICO = 0.35        # % de tokens numéricos para marcar una página como tabla
UMBRAL_NUM_MIXTO = 0.15
MAX_PUNTOS_X100 = 1.5         # puntos finales por cada 100 palabras (la prosa tiene 4-8)
UMBRAL_REPETIDO = 0.10        # bloque en el borde de ≥10 % de las páginas = encabezado o pie

# ------------------------------------------------------------------ RAG (clases S13_2, S14_1 y S14_2)
# Modelo multilingüe de embeddings. Se eligió sobre all-MiniLM-L6-v2 (el de clase) porque este
# último está entrenado sobre todo en inglés y recorta a 256 tokens; los párrafos largos no caben.
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
PREFIJO_DOCUMENTO = "passage: "   # los modelos E5 piden estos prefijos
PREFIJO_PREGUNTA = "query: "
K = 5                              # párrafos que se recuperan por pregunta
UMBRAL_SIMILITUD = 0.80            # PROVISIONAL: se calibra con el conjunto dev (notebook, sección 6)
MODELO_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # el mismo de la clase S14_1 / S14_2
