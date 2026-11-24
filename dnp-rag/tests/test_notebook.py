"""El notebook: (1) compila y solo llama a funciones que existen; (2) se ejecuta de punta a punta con datos y modelos de mentira."""
import ast
import importlib
import re
import shutil
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from dnp_rag import config, indice

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "notebooks" / "DNP_RAG_v5.ipynb"


def celdas_de_codigo():
    nb = nbformat.read(RUTA, as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            # las magias de Colab (!pip, %cd) no son Python: se cambian por `pass` conservando la sangría
            yield "\n".join(re.sub(r"^(\s*)[!%].*", r"\1pass", ln) for ln in c.source.splitlines())


def test_las_celdas_compilan():
    for src in celdas_de_codigo():
        ast.parse(src)


def test_el_notebook_se_entrega_sin_salidas():
    nb = nbformat.read(RUTA, as_version=4)
    assert all(not c.outputs and c.execution_count is None for c in nb.cells if c.cell_type == "code")


def test_las_funciones_del_paquete_que_llama_el_notebook_existen():
    arboles = [ast.parse(s) for s in celdas_de_codigo()]
    modulos = {}
    for t in arboles:
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module == "dnp_rag":
                for a in n.names:
                    modulos[a.asname or a.name] = importlib.import_module(f"dnp_rag.{a.name}")
    assert {"rag", "evaluacion", "recuperacion", "indice"} <= set(modulos)
    usados = 0
    for t in arboles:
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in modulos:
                assert hasattr(modulos[n.value.id], n.attr), f"{n.value.id}.{n.attr} no existe"
                usados += 1
    assert usados > 25


# ------------------------------------------------------------------ ejecución real de las celdas
def ejecutar(ns):
    for i, src in enumerate(celdas_de_codigo()):
        try:
            exec(compile(src, f"<celda {i}>", "exec"), ns)
        except Exception as e:
            raise AssertionError(f"Falló la celda {i}:\n{src[:300]}\n-> {type(e).__name__}: {e}") from e


@pytest.fixture()
def entorno(datos_tmp, corpus, embeddings, monkeypatch, tmp_path):
    """Corpus ya procesado + PDF de relleno (para no descargar) + gold en una carpeta temporal + embeddings falsos."""
    pytest.importorskip("IPython")
    (config.PDF_DIR / "x.pdf").write_bytes(b"%PDF-1.4 vacio")
    (config.CORPUS_DIR / "corpus_paginas.jsonl").write_text("")
    corpus.drop(columns=["tema", "tema_puntaje"]).to_csv(config.CORPUS_DIR / "parrafos.csv", index=False)
    gold = tmp_path / "gold"
    gold.mkdir()
    shutil.copy(RAIZ / "data" / "gold" / "preguntas_rag.csv", gold)
    shutil.copy(RAIZ / "data" / "gold" / "split.csv", gold)
    monkeypatch.setattr(config, "GOLD_DIR", gold)
    monkeypatch.setattr(indice, "crear_embeddings", lambda: embeddings)
    return gold


def test_el_notebook_corre_con_el_gold_incompleto_como_esta_hoy(entorno, capsys):
    ns = {}
    ejecutar(ns)
    out = capsys.readouterr().out
    assert ns["LISTO_DEV"] is False and ns["LISTO_TEST"] is False
    assert "Aún faltan documento y páginas" in out
    assert not (config.RESULTADOS_DIR / "resultados_test.csv").exists()        # sin gold completo, no se inventa ningún resultado
    assert "No encontré evidencia" in out or "¿Puedes darme más detalle?" in out    # la sección 7 sí corre


def test_el_notebook_corre_de_punta_a_punta_con_el_gold_completo(entorno, corpus, capsys):
    gold = pd.read_csv(entorno / "preguntas_rag.csv", dtype=str, keep_default_na=False)
    muestras = corpus.groupby("periodo").head(4).head(20)
    dominio = gold[gold["tipo"] == "dominio"].reset_index(drop=True)
    for i, (_, m) in enumerate(muestras.iterrows()):
        dominio.loc[i, ["pregunta", "documento", "paginas"]] = [" ".join(m["texto"].split()[-9:]), m["documento"], str(m["pagina"])]
    pd.concat([dominio, gold[gold["tipo"] == "fuera"]]).to_csv(entorno / "preguntas_rag.csv", index=False)
    (entorno / "split.csv").unlink()                                               # se vuelve a partir con las preguntas nuevas

    ns = {}
    ejecutar(ns)
    assert ns["LISTO_DEV"] is True and ns["LISTO_TEST"] is True
    assert set(ns["tabla_test"].index) == {"TF-IDF", "TF-IDF + gobierno", "Denso", "Denso + gobierno"}
    assert {"hit@1", "hit@5", "MRR", "IC95_hit@5"} <= set(ns["tabla_test"].columns)
    assert (config.RESULTADOS_DIR / "umbral.json").exists() and (config.RESULTADOS_DIR / "resultados_test.csv").exists()
    assert (entorno / "rubrica_respuestas.csv").exists()
    assert 0 <= ns["umbral"] <= 1
