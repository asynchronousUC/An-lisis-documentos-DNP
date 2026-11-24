"""Etapa 8: interfaz de chat con Gradio (clase S14_2). Es solo la "cara" del RAG de rag.py."""
from . import config, rag


def lanzar_chat(vs, embeddings, llm=None, compartir=False):
    """Abre un chat. Cada pregunta se responde de forma independiente (no recuerda la conversación)."""
    import gradio as gr

    def chatear(mensaje, historial):
        return rag.texto_chat(rag.responder(mensaje, vs, embeddings, llm=llm, k=config.K))

    demo = gr.ChatInterface(
        fn=chatear,
        title="Asistente de los Planes Nacionales de Desarrollo (2006-2026)",
        description="Responde con evidencia de los PND y cita gobierno, documento y página. "
                    "Si no encuentra evidencia, lo dice.",
        examples=["¿Qué metas de vivienda de interés social plantea el PND 2022-2026?",
                  "¿Qué dice el plan 2018-2022 sobre el catastro multipropósito?",
                  "¿Cómo se plantea reducir la informalidad laboral en el plan de Santos 2014-2018?"],
    )
    demo.launch(share=compartir)
    return demo
