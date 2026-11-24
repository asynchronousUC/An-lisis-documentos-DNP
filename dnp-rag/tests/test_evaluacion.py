import csv

import numpy as np
import pandas as pd
import pytest

from dnp_rag import evaluacion as ev

CSV_MAL_FORMADO = """pregunta,tipo,documento,paginas,notas
¿Qué plantea el plan sobre vivienda?,dominio,,,BORRADOR: verifica que exista; si no está, reemplaza la pregunta.
¿Cuál es la capital de Mongolia?,fuera,,,fuera de dominio
"""


def escribir(ruta, texto):
    ruta.write_text(texto, encoding="utf-8")
    return ruta


# ------------------------------------------------------------------ el gold que entrega el equipo
def test_el_gold_del_repositorio_se_lee_bien():
    """Todas las líneas tienen 5 campos y pandas no desplaza las columnas."""
    with open(ev.ruta_gold(), encoding="utf-8", newline="") as f:
        assert {len(r) for r in csv.reader(f)} == {5}
    gold = ev.cargar_gold()
    assert list(gold["tipo"].value_counts().sort_index()) == [20, 8]
    assert isinstance(gold.index, pd.RangeIndex) and gold["pregunta"].str.startswith("¿").all()


def test_el_split_del_repositorio_cubre_todo_el_gold_y_esta_balanceado():
    gold = ev.cargar_gold()
    split = pd.read_csv(ev.ruta_split(), dtype=str).set_index("pregunta")["split"]
    assert set(gold["pregunta"]) == set(split.index)
    g = gold.assign(split=gold["pregunta"].map(split))
    assert g.groupby(["split", "tipo"]).size().to_dict() == {("dev", "dominio"): 10, ("dev", "fuera"): 4,
                                                                ("test", "dominio"): 10, ("test", "fuera"): 4}


# ------------------------------------------------------------------ cargar_gold
def test_csv_con_comas_sin_comillas_falla_con_mensaje_claro(tmp_path):
    """Con pandas solo, la coma de `notas` desplaza TODAS las columnas sin avisar."""
    ruta = escribir(tmp_path / "g.csv", CSV_MAL_FORMADO)
    desplazado = pd.read_csv(ruta, dtype=str)
    assert "tipo" not in desplazado.columns or desplazado["tipo"].isna().all() or not desplazado["tipo"].isin(ev.TIPOS).any()
    with pytest.raises(ValueError, match="comillas"):
        ev.cargar_gold(ruta)


def test_validaciones_del_gold(tmp_path):
    with pytest.raises(ValueError, match="cabecera"):
        ev.cargar_gold(escribir(tmp_path / "a.csv", "pregunta,tipo\nx,dominio\n"))
    with pytest.raises(ValueError, match="solo puede ser"):
        ev.cargar_gold(escribir(tmp_path / "b.csv", "pregunta,tipo,documento,paginas,notas\nx,otro,,,\n"))
    with pytest.raises(ValueError, match="repetidas"):
        ev.cargar_gold(escribir(tmp_path / "c.csv", "pregunta,tipo,documento,paginas,notas\nx,dominio,,,\nx,fuera,,,\n"))


def test_pregunta_completa_solo_si_tiene_documento_y_paginas(tmp_path):
    ruta = escribir(tmp_path / "g.csv", "pregunta,tipo,documento,paginas,notas\n"
                    "a,dominio,tomo-1,120;121,\nb,dominio,tomo-1,,\nc,dominio,,45,\nd,dominio,tomo-1,abc,\ne,fuera,,,\n")
    assert ev.cargar_gold(ruta)["completa"].tolist() == [True, False, False, False, False]


def test_parsear_paginas():
    assert ev.parsear_paginas("120;121") == {120, 121}
    assert ev.parsear_paginas("120, 121") == {120, 121}
    assert ev.parsear_paginas("120-122") == {120, 121, 122}
    assert ev.parsear_paginas("45") == {45}
    assert ev.parsear_paginas("abc;;") == set() and ev.parsear_paginas("") == set()


# ------------------------------------------------------------------ split dev/test
def test_split_congelado_no_cambia_y_solo_asigna_las_nuevas(tmp_path):
    gold = pd.DataFrame({"pregunta": [f"d{i}" for i in range(10)] + [f"f{i}" for i in range(4)],
                         "tipo": ["dominio"] * 10 + ["fuera"] * 4})
    ruta = tmp_path / "split.csv"
    g1 = ev.crear_split(gold, ruta)
    assert g1.groupby(["split", "tipo"]).size().to_dict() == {("dev", "dominio"): 5, ("dev", "fuera"): 2,
                                                                ("test", "dominio"): 5, ("test", "fuera"): 2}
    assert ev.crear_split(gold, ruta)["split"].tolist() == g1["split"].tolist()          # repetir no cambia nada
    gold2 = pd.concat([gold, pd.DataFrame({"pregunta": ["nueva1", "nueva2"], "tipo": ["dominio", "dominio"]})], ignore_index=True)
    g2 = ev.crear_split(gold2, ruta)
    assert g2["split"].iloc[:14].tolist() == g1["split"].tolist()                         # las viejas conservan su partición
    assert set(g2["split"].iloc[14:]) == {"dev", "test"}


# ------------------------------------------------------------------ métricas
def gold_con_paginas():
    return pd.DataFrame({"pregunta": ["q1", "q2", "q3", "q4"], "tipo": ["dominio"] * 4,
                         "documento": ["tomo-1"] * 4, "paginas": ["10", "20", "30", "40"], "completa": [True] * 4})


def buscar_falso(pregunta, k, periodos):
    """q1 acierta en el puesto 1, q2 en el 3, q3 en una página vecina (39 vs 40... ver abajo), q4 no aparece."""
    tabla = {"q1": [("tomo-1", 10)], "q2": [("tomo-1", 1), ("tomo-1", 2), ("tomo-1", 20)],
             "q3": [("tomo-1", 31)], "q4": [("tomo-9", 40)]}
    return [{"documento": d, "pagina": p, "similitud": 0.9 - i * 0.1} for i, (d, p) in enumerate(tabla[pregunta][:k])]


def test_hit_y_mrr():
    r = ev.evaluar_recuperacion(buscar_falso, gold_con_paginas(), ks=(1, 3, 5))
    assert r["n"] == 4 and r["hit@1"] == 0.25 and r["hit@3"] == 0.5 and r["hit@5"] == 0.5
    assert r["MRR"] == pytest.approx((1 + 1 / 3) / 4, abs=1e-3)


def test_tolerancia_acepta_pagina_vecina():
    gold = gold_con_paginas()
    estricta = ev.evaluar_recuperacion(buscar_falso, gold, ks=(1, 5), tolerancia=0)["hit@5"]
    laxa = ev.evaluar_recuperacion(buscar_falso, gold, ks=(1, 5), tolerancia=1)["hit@5"]
    assert estricta == 0.5 and laxa == 0.75            # q3: esperada 30, devolvió 31


def test_detalle_marca_los_fallos():
    d = ev.detalle_recuperacion(buscar_falso, gold_con_paginas())
    assert d["rango_correcto"].iloc[:2].tolist() == [1, 3] and d["rango_correcto"].iloc[2:].isna().all()
    assert d["top1"].iloc[3] == "tomo-9 p.40"


def test_solo_cuentan_las_preguntas_completas():
    gold = gold_con_paginas()
    gold.loc[[1, 2, 3], "completa"] = False
    assert ev.evaluar_recuperacion(buscar_falso, gold)["n"] == 1


def test_wilson_es_ancho_con_pocas_preguntas():
    lo, hi = ev.wilson(8, 10)
    assert 0.45 < lo < 0.50 and 0.93 < hi < 0.95            # 8 de 10 -> [0,49 ; 0,94]
    assert ev.wilson(0, 0) == (0.0, 0.0)


def test_umbral_y_rechazo():
    dominio, fuera = np.array([0.9, 0.85, 0.8, 0.95]), np.array([0.5, 0.6, 0.7, 0.82])
    umbral, balanceada = ev.calibrar_umbral(dominio, fuera)
    assert 0.8 <= umbral <= 0.85 and balanceada >= 0.875
    r = ev.evaluar_rechazo(dominio, fuera, 0.8)
    assert r["rechaza_fuera_%"] == 75.0 and r["rechaza_dominio_por_error_%"] == 0.0


def test_rubrica_no_sobrescribe_calificaciones(tmp_path):
    resp = {"respuesta": "• algo [1]", "evidencias": [{"periodo": "2018-2022", "documento": "tomo-1", "pagina": 5}]}
    ruta = tmp_path / "rubrica.csv"
    ev.crear_hoja_rubrica([("q?", resp)], ruta)
    hoja = pd.read_csv(ruta)
    assert {"fidelidad_A", "citas_B", "notas"} <= set(hoja.columns) and "2018-2022" in hoja["fuentes"].iloc[0]
    hoja["fidelidad_A"], hoja["fidelidad_B"] = 2, 2
    hoja.to_csv(ruta, index=False)
    with pytest.raises(FileExistsError):
        ev.crear_hoja_rubrica([("q?", resp)], ruta)
    assert ev.resumir_rubrica(pd.read_csv(ruta)).set_index("criterio").loc["fidelidad", "misma_nota_%"] == 100.0
