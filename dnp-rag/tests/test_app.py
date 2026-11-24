import pytest

pytest.importorskip("gradio")

from dnp_rag import app, config, indice  # noqa: E402


def test_el_chat_se_construye_y_responde(corpus, embeddings, monkeypatch):
    """Se arma la interfaz de Gradio (sin abrirla) y se llama a su función de chat."""
    import gradio as gr

    lanzado = {}
    monkeypatch.setattr(gr.ChatInterface, "launch", lambda self, **kw: lanzado.update(kw))
    vs = indice.construir_indice(corpus, embeddings)
    monkeypatch.setattr(config, "UMBRAL_SIMILITUD", 0.3)

    demo = app.lanzar_chat(vs, embeddings, compartir=True)
    assert lanzado == {"share": True} and isinstance(demo, gr.ChatInterface)
    respuesta = demo.fn("¿Qué se propone sobre vivienda de interés social y subsidios habitacionales?", [])
    assert "Fuentes" in respuesta and "p." in respuesta
    assert "No encontré evidencia" in demo.fn("¿Cuál es la receta del sancocho con plátano y yuca?", [])
