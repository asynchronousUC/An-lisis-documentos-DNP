"""Descarga aparte (necesita internet): aquí se prueba extracción -> segmentación -> etiquetado con PDF sintéticos."""
import pandas as pd
import pymupdf
import pytest

from dnp_rag import config, diccionario, extraccion, segmentacion
from dnp_rag.descarga import filtrar, nombre_sin_ext

CUERPOS = [
    "El gobierno ampliará la cobertura de la educación escolar, fortalecerá a los docentes y reducirá la deserción de estudiantes en colegios y universidades de todo el territorio nacional durante el cuatrienio.",
    "Se promoverá el acceso a vivienda de interés social con subsidios habitacionales y crédito hipotecario, y se reducirá el déficit habitacional en las principales ciudades del país.",
    "La política de empleo buscará reducir el desempleo y la informalidad laboral y mejorar el mercado laboral de los trabajadores jóvenes mediante trabajo decente en todas las regiones.",
]


def crear_pdf(ruta, periodo, n_paginas=14):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for i in range(1, n_paginas + 1):
        pag = doc.new_page()
        pag.insert_textbox(pymupdf.Rect(40, 20, 560, 40), f"Plan Nacional de Desarrollo {periodo}", fontsize=8)   # encabezado repetido
        if i == 3:
            cuerpo = "Contenido\nCapítulo 1 ............ 5\nCapítulo 2 ............ 9\nCapítulo 3 ............ 14\nCapítulo 4 ............ 20"
        elif i == 5:
            cuerpo = "\n".join(" ".join(f"{(i * 7 + j * 3 + k) % 90 + 10},{k}" for k in range(8)) for j in range(14))
        else:
            cuerpo = f"{CUERPOS[i % 3]} Versión {i} del texto.\n\n{CUERPOS[(i + 1) % 3]} Anexo {i}."
        pag.insert_textbox(pymupdf.Rect(40, 60, 560, 760), cuerpo, fontsize=10)
    doc.save(ruta)
    doc.close()


def test_extraccion_segmentacion_y_etiquetado(datos_tmp):
    for periodo, presidente in (("2018-2022", "Iván Duque Márquez"), ("2022-2026", "Gustavo Petro")):
        crear_pdf(config.PDF_DIR / f"{periodo} - {presidente}" / f"tomo-{periodo}.pdf", periodo)

    reporte = extraccion.extraer_todo()
    assert reporte["paginas"].tolist() == [14, 14] and (reporte["pct_sin_texto"] == 0).all()
    paginas = pd.read_json(config.CORPUS_DIR / "corpus_paginas.jsonl", lines=True)
    assert len(paginas) == 28 and set(paginas["presidente"]) == {"Iván Duque Márquez", "Gustavo Petro"}
    assert not paginas["texto"].str.contains("Plan Nacional de Desarrollo 2018").any()      # el encabezado repetido se quitó
    assert paginas["pagina"].min() == 1                                                       # página FÍSICA, desde 1

    df, resumen = segmentacion.segmentar_todo()
    excluidas = pd.read_csv(config.CORPUS_DIR / "paginas_excluidas.csv")
    assert {"indice", "tabla"} <= set(excluidas["motivo"])
    assert df["id"].is_unique and (df["n_palabras"] >= config.MIN_PALABRAS).all()
    assert set(df["periodo"]) == {"2018-2022", "2022-2026"} and 3 not in set(df["pagina"]) and 5 not in set(df["pagina"])

    et = diccionario.etiquetar(df)
    assert {"Educación", "Vivienda", "Empleo"} <= set(et["tema"])


def test_extraer_sin_pdfs_da_un_mensaje_claro(datos_tmp):
    with pytest.raises(FileNotFoundError, match="descargar_todo"):
        extraccion.extraer_todo()


def test_seleccion_de_pdfs():
    res = [{"gobierno": "2014-2018 - Santos", "url_pdf": "https://x/CDT/PND/PND%202014-2018%20Tomo%201%20internet.pdf"},
           {"gobierno": "2014-2018 - Santos", "url_pdf": "https://x/CDT/OTSI/Certificado_de_Accesibilidad.pdf"}]
    assert nombre_sin_ext(res[0]["url_pdf"]) == "PND 2014-2018 Tomo 1 internet"
    elegidos, faltan = filtrar(res, {"2014-2018": ["PND 2014-2018 Tomo 1 internet", "PND 2014-2018 Tomo 2 internet"]})
    assert len(elegidos) == 1 and faltan == [("2014-2018", "PND 2014-2018 Tomo 2 internet")]
