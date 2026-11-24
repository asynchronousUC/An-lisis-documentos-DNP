"""Etapa 3: partir cada página en párrafos (segmentos) y descartar índices, tablas y ruido (v4)."""
import math
import re
from collections import Counter

import pandas as pd

from . import config

NUM_RE = re.compile(r"^[\(\$]?\d[\d.,%/\-–)]*$")                  # token numérico (cifras, %, rangos)
SOLO_NUM = re.compile(r"^(?:p[áa]g(?:ina)?\.?\s*)?(?:\d{1,4}|[ivxlcdm]{1,6})(?:\s*(?:de|/)\s*\d+)?$", re.I)
LIDER = re.compile(r"(?:\.\s?){4,}|…{2,}")                         # puntos guía de índices ".....12"
HEAD_IDX = re.compile(
    r"^\s*(índice|indice|contenido|contenidos|tabla de contenidos?|"
    r"(lista|índice) de (tablas|gráficos|graficos|figuras|cuadros|mapas|siglas))\b", re.I)
ABREV = (r"(?:Dra?|Sra?|No|Nº|N°|núm|Arts?|arts?|págs?|pp|Fig|Cap|cap|vs|Vol|vol|Ed|ed|Av|Cra|Cll|"
         r"Inc|Ltda|EE|UU|aprox|ej|Mg|Ing|Lic|Prof|Gral|Cel|Dir|Min)")   # abreviaturas: su punto no cierra oración


def bloques(texto):
    """Cada línea no vacía es un bloque."""
    return [b.strip() for b in re.split(r"\n+", texto) if b.strip()]


def normalizar(b):
    """Minúsculas y dígitos -> '#': así 'Página 12' y 'Página 13' cuentan como el mismo encabezado."""
    return re.sub(r"\d+", "#", b.lower()).strip()


def metricas(texto):
    """(n.º de palabras, proporción de tokens numéricos, puntos finales por cada 100 palabras)."""
    palabras = texto.split()
    n = len(palabras)
    if n == 0:
        return 0, 0.0, 0.0
    r_num = sum(bool(NUM_RE.match(p)) for p in palabras) / n
    puntos = len(re.findall(r"[.!?](?:\s|$)", texto)) * 100 / n
    return n, r_num, puntos


def es_tabla(texto):
    """Heurística: muchos números y pocas oraciones, o un rótulo 'Tabla/Cuadro/Gráfico n'."""
    n, r_num, puntos = metricas(texto)
    if n < 15:
        return False
    rotulo = bool(re.search(r"\b(Tabla|Cuadro|Gráfico|Grafico)\s+\d", texto[:300]))
    return (r_num >= config.UMBRAL_NUMERICO
            or (r_num >= config.UMBRAL_NUM_MIXTO and puntos < config.MAX_PUNTOS_X100)
            or (rotulo and r_num >= 0.10 and puntos < 3))


def es_indice(texto):
    """Índice: ≥ 3 puntos guía, o encabezado de índice y al menos uno."""
    bl = bloques(texto)
    if not bl:
        return False
    lideres = len(LIDER.findall(texto))
    return lideres >= 3 or (bool(HEAD_IDX.match(bl[0])) and lideres >= 1)


def bloques_repetidos(paginas):
    """Bloques cortos que aparecen en los bordes de muchas páginas (encabezados y pies)."""
    if len(paginas) < 10:
        return set()
    cont = Counter()
    for _, t in paginas:
        bl = bloques(t)
        for b in {normalizar(x) for x in bl[:2] + bl[-2:] if len(x.split()) < 15}:
            cont[b] += 1
    return {b for b, c in cont.items() if b and c / len(paginas) >= config.UMBRAL_REPETIDO}


def separar_oraciones(texto):
    """Divide en oraciones protegiendo abreviaturas e iniciales (J. M. Santos)."""
    t = re.sub(rf"\b({ABREV})\.", r"\1§", texto)
    t = re.sub(r"\b([A-ZÁÉÍÓÚÑ])\.(?=\s)", r"\1§", t)
    partes = re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑ¿¡«“\"(\[\d])", t)
    return [p.replace("§", ".").strip() for p in partes if p.strip()]


def dividir_por_oraciones(texto):
    """Parte un párrafo largo en trozos parejos (< MAX_CARACTERES) cortando en límites de oración."""
    if len(texto) <= config.MAX_CARACTERES:
        return [texto]
    n = math.ceil(len(texto) / config.MAX_CARACTERES)
    objetivo = len(texto) / n
    trozos, actual = [], ""
    for o in separar_oraciones(texto):
        candidato = f"{actual} {o}".strip()
        if actual and (len(candidato) > config.MAX_CARACTERES or len(actual) >= objetivo):
            trozos.append(actual)
            actual = o
        else:
            actual = candidato
    if actual:
        trozos.append(actual)
    # Un trozo final muy corto se pega al anterior. Primero se saca con pop() y DESPUÉS se pega:
    # escribir `trozos[-2] += " " + trozos.pop()` falla (IndexError) o pierde texto.
    if len(trozos) > 1 and len(trozos[-1].split()) < config.MIN_PALABRAS:
        ultimo = trozos.pop()
        trozos[-1] += " " + ultimo
    return trozos


def unir_continuaciones(items):
    """Un bloque que empieza en minúscula sigue al anterior (se cortó por salto de línea o de página)."""
    out = []
    for it in items:
        if out and it["texto"][:1].islower():
            out[-1]["texto"] += " " + it["texto"]
        else:
            out.append(dict(it))
    return out


def segmentar_documento(paginas):
    """paginas: lista de (n.º de página, texto). Devuelve (segmentos, páginas excluidas)."""
    repetidos = bloques_repetidos(paginas)
    items, excluidas = [], []
    for num, texto in paginas:
        if not texto:
            continue
        motivo = "indice" if es_indice(texto) else "tabla" if es_tabla(texto) else None
        if motivo:
            excluidas.append({"pagina": num, "motivo": motivo, "vista_previa": texto[:120].replace("\n", " ")})
            continue
        for b in bloques(texto):
            if len(b.split()) < 15 and normalizar(b) in repetidos:
                continue                                  # encabezado o pie de página
            if SOLO_NUM.match(b) or LIDER.search(b):
                continue                                  # número de página suelto o línea de índice
            items.append({"pagina": num, "texto": b})

    items = unir_continuaciones(items)
    items = [it for it in items if len(it["texto"].split()) >= config.MIN_PALABRAS]

    segmentos = []
    for it in items:
        for t in dividir_por_oraciones(it["texto"]):
            segmentos.append({"pagina": it["pagina"], "n_palabras": len(t.split()), "texto": t})
    return segmentos, excluidas


def segmentar_todo():
    """Lee corpus_paginas.jsonl y guarda parrafos.csv (un párrafo por fila) y paginas_excluidas.csv."""
    ruta = config.CORPUS_DIR / "corpus_paginas.jsonl"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Primero ejecuta extraccion.extraer_todo()")
    paginas = pd.read_json(ruta, lines=True)

    filas, excluidas, resumen = [], [], []
    for (periodo, presidente, documento), g in paginas.groupby(["periodo", "presidente", "documento"]):
        g = g.sort_values("pagina")
        segs, exc = segmentar_documento([(int(p), t or "") for p, t in zip(g["pagina"], g["texto"])])
        filas += [{"periodo": periodo, "presidente": presidente, "documento": documento, **s} for s in segs]
        excluidas += [{"periodo": periodo, "documento": documento, **e} for e in exc]
        resumen.append({"periodo": periodo, "documento": documento, "paginas": len(g),
                        "paginas_excluidas": len(exc), "segmentos": len(segs)})
        print(f"✔ {periodo} | {documento}: {len(segs)} segmentos, {len(exc)} páginas excluidas")

    df = pd.DataFrame(filas)
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(config.CORPUS_DIR / "parrafos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluidas).to_csv(config.CORPUS_DIR / "paginas_excluidas.csv", index=False, encoding="utf-8-sig")
    return df, pd.DataFrame(resumen)
