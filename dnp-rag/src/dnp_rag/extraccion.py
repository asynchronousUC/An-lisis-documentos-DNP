"""Etapa 2: sacar el texto de cada página de los PDF con PyMuPDF (viene de la v4)."""
import json
import re
from collections import Counter

import pandas as pd

from . import config


def limpiar_texto(t):
    """Quita guiones suaves, une palabras partidas y compacta espacios."""
    t = t.replace("\u00ad", "")
    t = re.sub(r"-\n(?=[a-záéíóúñü])", "", t)              # palabras cortadas con guion al final de línea
    t = re.sub(r"(?<![.:;!?\n])\n(?!\n)", " ", t)          # une líneas del mismo párrafo
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def lineas_repetidas(paginas_texto, umbral=0.4):
    """Líneas cortas que se repiten en muchas páginas: encabezados y pies de página."""
    n = len(paginas_texto)
    if n < 5:
        return set()
    cont = Counter()
    for t in paginas_texto:
        for linea in {x.strip() for x in t.split("\n") if 0 < len(x.strip()) < 80}:
            cont[linea] += 1
    return {linea for linea, c in cont.items() if c / n >= umbral}


def parse_gobierno(nombre_carpeta):
    """'2018-2022 - Iván Duque Márquez' -> ('2018-2022', 'Iván Duque Márquez')."""
    m = re.match(r"(\d{4}-\d{4})\s*-\s*(.+)", nombre_carpeta)
    return (m.group(1), m.group(2).strip()) if m else (nombre_carpeta, "")


def extraer_pdf(pdf_path):
    """Devuelve (registros por página, resumen de calidad) de un PDF."""
    import pymupdf   # PyMuPDF (en versiones antiguas: import fitz)

    periodo, presidente = parse_gobierno(pdf_path.parent.name)
    with pymupdf.open(pdf_path) as doc:
        crudo = [p.get_text("text", sort=True) for p in doc]   # sort=True: orden de lectura

    repetidas = lineas_repetidas(crudo)
    registros, sin_texto = [], 0
    for i, t in enumerate(crudo, 1):
        t = "\n".join(x for x in t.split("\n") if x.strip() not in repetidas)
        t = limpiar_texto(t)
        sin_texto += len(t) < config.MIN_CARACTERES_PAGINA
        # OJO: `pagina` es la página FÍSICA del PDF (1 = primera hoja), no el número impreso al pie.
        # Es la misma que muestra el visor de PDF y la que se anota en data/gold.
        registros.append({"periodo": periodo, "presidente": presidente, "documento": pdf_path.stem,
                          "pagina": i, "texto": t})
    resumen = {"periodo": periodo, "documento": pdf_path.stem, "paginas": len(crudo),
               "paginas_sin_texto": sin_texto, "pct_sin_texto": round(100 * sin_texto / max(len(crudo), 1), 1)}
    return registros, resumen


def extraer_todo():
    """Procesa todos los PDF de config.PDF_DIR y guarda corpus_paginas.jsonl y reporte_calidad.csv."""
    pdfs = sorted(config.PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDF en {config.PDF_DIR}. Primero ejecuta descarga.descargar_todo()")
    todos, reporte = [], []
    for pdf in pdfs:
        registros, resumen = extraer_pdf(pdf)
        todos += registros
        reporte.append(resumen)
        print(f"✔ {resumen['periodo']} | {pdf.name} | {resumen['paginas']} págs | {resumen['paginas_sin_texto']} sin texto")

    with open(config.CORPUS_DIR / "corpus_paginas.jsonl", "w", encoding="utf-8") as f:
        for r in todos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(reporte)
    df.to_csv(config.CORPUS_DIR / "reporte_calidad.csv", index=False, encoding="utf-8-sig")
    return df
