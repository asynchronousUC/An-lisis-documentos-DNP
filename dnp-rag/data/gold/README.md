# Gold: preguntas anotadas por el equipo

Este es la construccion de preguntas y es la referencia del proyecto (ver `docs/METODOLOGIA.md`).

## Archivos

| Archivo | Qué es |
|---|---|
| `preguntas_rag.csv` | las preguntas y dónde está la respuesta |
| `split.csv` | a cuál partición (`dev` o `test`) va cada pregunta |
| `rubrica_respuestas.csv` | calificación de las respuestas (se crea desde el notebook) |

## Columnas de `preguntas_rag.csv`

| Columna | Qué poner |
|---|---|
| `pregunta` | la pregunta, como la haría un usuario |
| `tipo` | `dominio` (los PND la responden) o `fuera` (no la responden) |
| `documento` | *solo dominio.* Una parte del nombre del PDF (sin importar mayúsculas), p. ej. `tomo-1-dnp` |
| `paginas` | *solo dominio.* La página **física** del PDF, la que muestra el visor (la primera hoja es la 1, no el número impreso al pie): `120`, `120;121` o `120-122` |
| `notas` | observaciones |

Nombres de PDF (columna `documento`): 2022-2026 → `plan-nacional-de-desarrollo-2022-2026` · 2018-2022 → `tomo-1-dnp`, `tomo-2-pnd` ·
2014-2018 → `PND 2014-2018 Tomo 1`, `PND 2014-2018 Tomo 2` · 2010-2014 → `Tomo I CD`, `Tomo II CD` · 2006-2010 → `PND_Tomo_1`, `PND_Tomo_2`.

El `documento` se busca como *subcadena* del nombre del archivo. Por eso se usa `Tomo I CD` y no `Tomo I`, que también calzaría con `Tomo II`.



## Cuidado al editar el CSV

- Guárdado como **CSV UTF-8 separado por comas** 
- Si un texto lleva comas, se pone **entre comillas dobles**. Una coma suelta en `notas` desplaza todas las columnas; `evaluacion.cargar_gold()` lo detecta y avisa con el número de línea.
- Para agregar preguntas, solo se añade filas.
- **No se cambian una pregunta de `test` después de ver los resultados.**
