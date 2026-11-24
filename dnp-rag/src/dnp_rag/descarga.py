"""Etapa 1: encontrar y descargar los PDF de los PND desde el sitio del DNP (viene de la v4)."""
import os
import re
import time
from urllib.parse import unquote, urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from requests.exceptions import SSLError

from . import config

sesion = requests.Session()
sesion.headers.update(config.HEADERS)


def limpiar(texto):
    """Quita caracteres invisibles (zero-width, BOM) que SharePoint mete en textos y enlaces."""
    return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", texto).strip()


def nombre_sin_ext(url):
    """Nombre del archivo de la URL, decodificado y sin la extensión .pdf."""
    return unquote(url.split("/")[-1]).rsplit(".pdf", 1)[0].strip()


def get(url, **kw):
    """GET con reintento sin verificar SSL, pero solo para el sitio del DNP.

    El servidor colaboracion.dnp.gov.co tiene la cadena de certificados incompleta
    (es el error SSLCertVerificationError que daba en la v4).
    """
    try:
        r = sesion.get(url, timeout=60, **kw)
    except SSLError:
        if "dnp.gov.co" not in url:
            raise
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = sesion.get(url, timeout=60, verify=False, **kw)
    r.raise_for_status()
    return r


def get_soup(url):
    return BeautifulSoup(get(url).text, "html.parser")


def buscar_gobiernos(periodos):
    """Enlaces (texto, url) de los gobiernos de la sección 'Históricos' + el PND vigente."""
    patron = re.compile(r"^(\d{4})\s*-\s*(\d{4})\s*-\s*")
    # el PND vigente no aparece en "Históricos": se agrega a mano
    gobiernos = [("2022-2026 - Gustavo Petro", config.URL_PND_ACTUAL)]
    for a in get_soup(config.URL_BASE).find_all("a", href=True):
        texto = re.sub(r"\s+", " ", limpiar(a.get_text(" ", strip=True)))
        if patron.match(texto):
            gobiernos.append((texto, urljoin(config.URL_BASE, limpiar(a["href"]))))
    gobiernos = list(dict.fromkeys(gobiernos))          # quita duplicados conservando el orden
    return [(g, u) for g, u in gobiernos if g[:9] in periodos]


def buscar_pdfs(gobiernos):
    """Para cada gobierno, recoge los enlaces a archivos PDF de su página."""
    resultados = []
    for gobierno, url_gob in gobiernos:
        try:
            soup = get_soup(url_gob)
        except Exception as e:
            print(f"[ERROR] {gobierno}: {e}")
            continue
        vistos = set()
        for a in soup.find_all("a", href=True):
            href = limpiar(a["href"])
            if ".pdf" not in href.lower():
                continue
            url_pdf = urljoin(url_gob, href)
            if url_pdf not in vistos:
                vistos.add(url_pdf)
                resultados.append({"gobierno": gobierno, "url_pdf": url_pdf})
        print(f"{gobierno}: {len(vistos)} PDF(s)")
        time.sleep(1)                                    # pausa para no saturar el servidor
    return resultados


def filtrar(resultados, seleccion):
    """Se queda con los PDF cuyo nombre está en `seleccion`. Devuelve (elegidos, no_encontrados)."""
    elegidos, encontrados = [], set()
    for r in resultados:
        periodo = r["gobierno"][:9]
        nombre = nombre_sin_ext(r["url_pdf"])
        for buscado in seleccion.get(periodo, []):
            if nombre.lower() == buscado.lower() and (periodo, buscado) not in encontrados:
                elegidos.append(r)
                encontrados.add((periodo, buscado))
    faltan = [(p, n) for p, ns in seleccion.items() for n in ns if (p, n) not in encontrados]
    return elegidos, faltan


def descargar_pdf(url, ruta):
    """Descarga por bloques y comprueba que el archivo sea un PDF (empieza con %PDF)."""
    r = get(url, stream=True)
    with open(ruta, "wb") as f:
        for bloque in r.iter_content(chunk_size=1024 * 64):
            f.write(bloque)
    with open(ruta, "rb") as f:
        if not f.read(5).startswith(b"%PDF"):
            os.remove(ruta)
            raise ValueError("El archivo descargado no es un PDF válido")


def descargar_todo(periodos=None):
    """Descarga los PDF de `periodos` (por defecto, todos los de config.SELECCION).

    Para una prueba rápida: descargar_todo(["2022-2026"]).
    Los PDF que ya existen no se vuelven a bajar.
    """
    periodos = periodos or config.PERIODOS
    seleccion = {p: config.SELECCION[p] for p in periodos}
    gobiernos = buscar_gobiernos(set(periodos))
    print("Gobiernos a revisar:", [g for g, _ in gobiernos])

    elegidos, faltan = filtrar(buscar_pdfs(gobiernos), seleccion)
    for periodo, nombre in faltan:
        print(f"[NO ENCONTRADO] {periodo} -> {nombre}")

    fallidos = 0
    for i, r in enumerate(elegidos, 1):
        carpeta = config.PDF_DIR / re.sub(r'[\\/:*?"<>|]', "", r["gobierno"])
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / re.sub(r'[\\/:*?"<>|]', "_", unquote(r["url_pdf"].split("/")[-1]))
        if ruta.exists():
            print(f"[{i}/{len(elegidos)}] ya existe: {ruta.name}")
            continue
        try:
            descargar_pdf(r["url_pdf"], ruta)
            print(f"[{i}/{len(elegidos)}] OK  {ruta.name}")
        except Exception as e:
            fallidos += 1
            print(f"[{i}/{len(elegidos)}] ERROR {r['url_pdf']}\n    {e}")
        time.sleep(1)
    print(f"\nDescargados: {len(elegidos) - fallidos} | Fallidos: {fallidos} | Carpeta: {config.PDF_DIR}")
