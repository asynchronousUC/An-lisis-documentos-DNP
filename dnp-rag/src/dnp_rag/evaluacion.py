"""Etapa 7: evaluación INDEPENDIENTE contra las preguntas anotadas por personas (data/gold).

Por qué existe: en la v4 las categorías salían de un diccionario de palabras clave y luego un modelo
aprendía a predecirlas; una exactitud alta solo mostraba que el modelo repetía la regla. Aquí la verdad
de referencia (gold) la ponen personas, y es independiente de cualquier diccionario o modelo:

  - Cada pregunta del dominio trae el DOCUMENTO y las PÁGINAS donde está la respuesta, buscadas por una
    persona en el PDF (no a partir de lo que devuelve el sistema).
  - Las preguntas "fuera" son de temas que los PND no tratan: el sistema debe decir que no hay evidencia.
  - El gold se parte UNA vez en `dev` y `test` (data/gold/split.csv, congelado). Con `dev` se toman todas las
    decisiones (k, umbral, qué recuperador usar). `test` se mira solo al final, para el informe.
"""
import csv
import re

import numpy as np
import pandas as pd

from . import config
from .recuperacion import detectar_periodos

COLUMNAS_GOLD = ["pregunta", "tipo", "documento", "paginas", "notas"]
TIPOS = ("dominio", "fuera")


def ruta_gold():
    """Se resuelve al usarla (no al importar), por si se cambia config.GOLD_DIR."""
    return config.GOLD_DIR / "preguntas_rag.csv"


def ruta_split():
    return config.GOLD_DIR / "split.csv"


# ================================================================== 1) cargar el gold
def parsear_paginas(valor):
    """'120;121', '120, 121', '120-122' o '45' -> {120, 121, ...}. Lo que no sea un número se ignora."""
    paginas = set()
    for token in re.split(r"[;,\s]+", str(valor).strip()):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if m:
            paginas.update(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
    return paginas


def cargar_gold(ruta=None):
    """Lee preguntas_rag.csv y lo valida. Falla con un mensaje claro si el archivo está mal formado.

    Con pandas solo, un texto con comas sin comillas (p. ej. en `notas`) desplaza TODAS las columnas sin
    avisar y la evaluación sale mal. Por eso primero se cuentan los campos de cada línea.
    """
    ruta = ruta or ruta_gold()
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = list(csv.reader(f))
    if filas[0] != COLUMNAS_GOLD:
        raise ValueError(f"{ruta}: la cabecera debe ser {COLUMNAS_GOLD}, pero es {filas[0]}")
    malas = [i for i, r in enumerate(filas, 1) if len(r) != len(COLUMNAS_GOLD)]
    if malas:
        raise ValueError(f"{ruta}: las líneas {malas} no tienen {len(COLUMNAS_GOLD)} campos. Seguramente hay una coma "
                         "dentro de un texto: pon ese texto entre comillas dobles (o quita la coma).")

    gold = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    gold = gold.apply(lambda col: col.str.strip())
    malos = gold.loc[~gold["tipo"].isin(TIPOS), "tipo"].unique()
    if len(malos):
        raise ValueError(f"{ruta}: 'tipo' solo puede ser {TIPOS}; hay {list(malos)}")
    if gold["pregunta"].duplicated().any():
        raise ValueError(f"{ruta}: hay preguntas repetidas: {gold.loc[gold['pregunta'].duplicated(), 'pregunta'].tolist()}")
    # Una pregunta del dominio sirve para medir la recuperación solo si ya tiene documento y páginas
    gold["completa"] = (gold["tipo"] == "dominio") & (gold["documento"] != "") & gold["paginas"].map(bool) \
        & gold["paginas"].map(lambda p: len(parsear_paginas(p)) > 0)
    return gold


def resumen_gold(gold):
    """Cuántas preguntas hay (por tipo y partición) y cuántas del dominio ya tienen documento y páginas."""
    cols = ["split", "tipo"] if "split" in gold else ["tipo"]
    t = gold.groupby(cols).agg(preguntas=("pregunta", "size"), listas_para_evaluar=("completa", "sum"))
    t.loc[t.index.get_level_values("tipo") == "fuera", "listas_para_evaluar"] = None   # las de fuera no necesitan páginas
    return t


# ================================================================== 2) dev / test congelado
def crear_split(gold, ruta=None, test_frac=0.5, semilla=config.SEMILLA):
    """Agrega la columna `split` ('dev' o 'test'). La partición se guarda y NO cambia entre ejecuciones.

    Si se agregan preguntas nuevas al gold, solo ellas reciben partición; las que ya tenían la conservan.
    La partición se hace por tipo, para que dev y test tengan preguntas de dominio y de fuera de dominio.
    """
    ruta = ruta or ruta_split()
    previo = pd.read_csv(ruta, dtype=str) if ruta.exists() else pd.DataFrame(columns=["pregunta", "split"])
    asignado = dict(zip(previo["pregunta"], previo["split"]))
    rng = np.random.default_rng(semilla)
    for tipo in TIPOS:
        nuevas = [p for p in gold.loc[gold["tipo"] == tipo, "pregunta"] if p not in asignado]
        rng.shuffle(nuevas)
        n_test = int(round(len(nuevas) * test_frac))
        for i, p in enumerate(nuevas):
            asignado[p] = "test" if i < n_test else "dev"
    pd.DataFrame({"pregunta": list(asignado), "split": list(asignado.values())}).to_csv(ruta, index=False, encoding="utf-8")
    return gold.assign(split=gold["pregunta"].map(asignado))


# ================================================================== 3) métricas de recuperación
def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza del 95 % para una proporción. Con pocas preguntas es MUY ancho."""
    if n == 0:
        return 0.0, 0.0
    p = aciertos / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(centro - mitad), float(centro + mitad)


def _rango_correcto(buscar, fila, k_max, usar_periodo, tolerancia):
    """Posición (1, 2, ...) del primer resultado que cae en el documento y las páginas esperadas; None si no aparece."""
    periodos = detectar_periodos(fila["pregunta"]) if usar_periodo else None
    esperadas = parsear_paginas(fila["paginas"])
    validas = {p + d for p in esperadas for d in range(-tolerancia, tolerancia + 1)}
    for i, ev in enumerate(buscar(fila["pregunta"], k_max, periodos), 1):
        if fila["documento"].lower() in ev["documento"].lower() and ev["pagina"] in validas:
            return i
    return None


def evaluar_recuperacion(buscar, gold, ks=(1, 3, 5), usar_periodo=True, tolerancia=0):
    """hit@k y MRR sobre las preguntas del dominio que ya tienen documento y páginas.

    `buscar(pregunta, k, periodos)` devuelve una lista de diccionarios con `documento` y `pagina`
    (ver recuperacion.buscar_denso / buscar_tfidf). `tolerancia=1` acepta una página vecina (útil porque un
    párrafo que cruza de una hoja a otra queda registrado en la primera).
    """
    d = gold[gold["completa"]]
    rangos = [_rango_correcto(buscar, f, max(ks), usar_periodo, tolerancia) for _, f in d.iterrows()]
    n = len(rangos)
    res = {"n": n}
    for k in ks:
        aciertos = sum(r is not None and r <= k for r in rangos)
        res[f"hit@{k}"] = round(aciertos / n, 3) if n else float("nan")
    res["MRR"] = round(float(np.mean([1 / r if r else 0.0 for r in rangos])), 3) if n else float("nan")
    k5 = 5 if 5 in ks else max(ks)
    lo, hi = wilson(sum(r is not None and r <= k5 for r in rangos), n)
    res[f"IC95_hit@{k5}"] = f"[{lo:.2f}, {hi:.2f}]"
    return res


def detalle_recuperacion(buscar, gold, k_max=10, usar_periodo=True, tolerancia=0):
    """Por pregunta: en qué posición apareció la respuesta correcta y qué devolvió el sistema primero.

    Es la tabla para entender los fallos (¿página vecina?, ¿otro gobierno?, ¿la respuesta está en una tabla?).
    """
    filas = []
    for _, f in gold[gold["completa"]].iterrows():
        periodos = detectar_periodos(f["pregunta"]) if usar_periodo else None
        evs = buscar(f["pregunta"], k_max, periodos)
        filas.append({
            "pregunta": f["pregunta"], "esperado": f"{f['documento']} p.{f['paginas']}",
            "rango_correcto": _rango_correcto(buscar, f, k_max, usar_periodo, tolerancia),
            "top1": f"{evs[0]['documento']} p.{evs[0]['pagina']}" if evs else "(sin resultados)",
            "similitud_top1": round(evs[0]["similitud"], 3) if evs else 0.0})
    return pd.DataFrame(filas)


# ================================================================== 4) umbral «no hay evidencia»
def similitudes_maximas(buscar, preguntas, usar_periodo=True):
    """Similitud del mejor resultado de cada pregunta (con el recuperador denso, que es el que usa el umbral)."""
    out = []
    for q in preguntas:
        evs = buscar(q, 1, detectar_periodos(q) if usar_periodo else None)
        out.append(evs[0]["similitud"] if evs else 0.0)
    return np.array(out)


def calibrar_umbral(sim_dominio, sim_fuera):
    """Umbral que mejor separa preguntas del dominio (similitud alta) de las de fuera (baja).

    Maximiza la exactitud balanceada. SE CALIBRA SOLO CON DEV. Devuelve (umbral, exactitud balanceada).
    """
    sim_dominio, sim_fuera = np.asarray(sim_dominio, float), np.asarray(sim_fuera, float)
    mejor = (0.0, 0.0)
    for t in np.unique(np.concatenate([sim_dominio, sim_fuera])):
        balanceada = ((sim_dominio >= t).mean() + (sim_fuera < t).mean()) / 2
        if balanceada > mejor[1]:
            mejor = (float(t), float(balanceada))
    return mejor


def evaluar_rechazo(sim_dominio, sim_fuera, umbral):
    """Con un umbral: % de preguntas fuera de dominio rechazadas (debe ser alto) y de dominio rechazadas por error (bajo)."""
    return {"umbral": round(float(umbral), 3),
            "rechaza_fuera_%": round(100 * float((np.asarray(sim_fuera) < umbral).mean()), 1) if len(sim_fuera) else None,
            "rechaza_dominio_por_error_%": round(100 * float((np.asarray(sim_dominio) < umbral).mean()), 1) if len(sim_dominio) else None,
            "n_dominio": len(sim_dominio), "n_fuera": len(sim_fuera)}


# ================================================================== 5) calidad de las respuestas (a mano)
CRITERIOS = ("fidelidad", "relevancia", "citas")


def crear_hoja_rubrica(resultados, ruta, forzar=False):
    """CSV para que dos personas califiquen de 0 a 2 cada respuesta.

    fidelidad: ¿todo lo que afirma está en las fuentes? · relevancia: ¿responde lo que se preguntó? ·
    citas: ¿cada [n] apunta a la fuente correcta?  (0 = no, 1 = en parte, 2 = sí)
    `resultados`: lista de (pregunta, respuesta_de_rag.responder). No sobrescribe una hoja ya calificada.
    """
    if ruta.exists() and not forzar:
        raise FileExistsError(f"{ruta} ya existe y puede tener calificaciones. Usa forzar=True para rehacerla.")
    filas = []
    for pregunta, resp in resultados:
        fuentes = " | ".join(f"[{i}] {e['periodo']} {e['documento']} p.{e['pagina']}" for i, e in enumerate(resp["evidencias"], 1))
        filas.append({"pregunta": pregunta, "respuesta": resp["respuesta"], "fuentes": fuentes,
                      **{f"{c}_{p}": "" for p in ("A", "B") for c in CRITERIOS}, "notas": ""})
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")


def resumir_rubrica(hoja):
    """Promedio por criterio y % de respuestas en que las dos personas dieron la misma nota."""
    filas = []
    for c in CRITERIOS:
        a = pd.to_numeric(hoja[f"{c}_A"], errors="coerce")
        b = pd.to_numeric(hoja[f"{c}_B"], errors="coerce")
        ok = a.notna() & b.notna()
        filas.append({"criterio": c, "media_A": round(a.mean(), 2), "media_B": round(b.mean(), 2),
                      "misma_nota_%": round(100 * float((a[ok] == b[ok]).mean()), 1) if ok.any() else None, "n": int(ok.sum())})
    return pd.DataFrame(filas)
