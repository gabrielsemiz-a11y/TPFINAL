# Trabajo final — Arquitectura de IA, ML y gobernanza

Resolución explicada de los tres ejercicios y referencia para el simulador educativo del repositorio. Los diagramas usan formas de diagrama de flujo convencionales: **óvalo** = inicio/fin; **paralelogramo** = entrada/salida; **rectángulo** = proceso; **rombo** = decisión. Los íconos 🧠 identifican un LLM u orquestador, no un componente de control de seguridad.

> Las decisiones de diseño y el dataset del simulador son didácticos. No representan asesoramiento fiscal, un sistema productivo ni una certificación ISO.
>
> ## Ejercicio 1 — Diagnóstico de desviaciones y métricas
>
> ### Aprendizaje
> Es **aprendizaje supervisado de regresión**: se predice una magnitud continua, el consumo eléctrico diario. Para aprender, cada ejemplo histórico necesita variables de entrada (temperatura, estación, características del hogar y hábitos) junto con el consumo real medido como etiqueta. El algoritmo ajusta sus parámetros para reducir la diferencia entre el valor real y el estimado. No es clustering no supervisado ni un agente que optimiza acciones mediante recompensas.
>
> ### Diagnóstico y respuesta
> El cambio de estación provoca **deriva de datos (data drift)**: las entradas de invierno dejan de parecerse a las de verano. Si cambia la distribución de las variables de entrada puede hablarse de *covariate shift*; si también cambia la relación entre entradas y consumo —por ejemplo, calefacción frente a refrigeración— hay *concept drift*. El código y la infraestructura pueden permanecer iguales y, aun así, caer la calidad porque el modelo recibe condiciones que no aprendió bien.
>
> Solución dentro del ciclo de vida del dato: (1) monitorear distribuciones y calidad de entradas por estación; (2) recopilar y validar consumos reales de invierno y verano; (3) actualizar el dataset representativo; (4) reentrenar o recalibrar **el mismo tipo de algoritmo**, validar con datos temporales no vistos y desplegar solo si supera el umbral acordado. No basta con cambiar etiquetas sin comprobar representatividad. En producción, las métricas supervisadas requieren esperar o asociar cada predicción con su consumo real.
>
> ### Métricas
> - **MAE**: error absoluto medio en kWh; comunicación simple del error típico y menos sensible a valores extremos.
> - - **RMSE**: raíz del error cuadrático medio, también en kWh; penaliza más los errores grandes, apropiado para vigilar fallas drásticas. Si RMSE supera mucho al MAE, revisar errores extremos.
>   - - **R²**: proporción de variabilidad explicada respecto de una referencia; complementa, pero no indica por sí solo cuántos kWh se erra y puede ser negativo.
>    
>     - **Recomendación:** tablero principal con MAE + RMSE por estación y segmento; R² como complemento. Ninguna métrica reemplaza validar residuos, calidad de datos y cambios de población.
>    
>     - ```mermaid
>       flowchart LR
>        A([Inicio]) --> B[/Entradas del hogar y estación/]
>        B --> C[Modelo supervisado de regresión]
>        C --> D[/Consumo estimado ŷ en kWh/]
>        E[/Consumo real y/ --> F[Calcular MAE, RMSE y R²]
>        D --> F --> G{¿Deriva o error fuera del umbral?}
>        G -- No --> H[Continuar monitoreo]
>        G -- Sí --> I[Recolectar y validar datos estacionales]
>        I --> J[Reentrenar mismo algoritmo y validar]
>        J --> K{¿Supera validación temporal?}
>        K -- Sí --> L[Desplegar versión aprobada]
>        K -- No --> M[Investigar datos y mantener versión estable]
>       ```
>
> ## Ejercicio 2 — Memoria conversacional y RAG
>
> ### Memoria
> Falló la **gestión de estado/memoria conversacional**: cada llamada trató la consulta como aislada o no recuperó el historial de la sesión. La ventana de contexto del LLM tampoco es una base de datos persistente. Mantener un manual de 5.000 páginas en cada prompt agrava costo y ruido, pero no equivale a memoria del chat.
>
> Usar un identificador de sesión y guardar turnos en almacenamiento persistente del servidor; al responder, recuperar solo los últimos turnos relevantes y, si la charla crece, un resumen actualizado con entidades, decisiones y pendientes. Separar la memoria de corto plazo del conocimiento documental (RAG), controlar vencimiento y privacidad, y permitir borrar/exportar el historial según la política del sistema. El prompt recibe el contexto acotado de esa conversación, no todo el registro ni el manual.
>
> ### Pipeline RAG
> 1. **Ingesta e indexación:** extraer texto y metadatos del PDF (título, página, vigencia); limpiar, dividir por secciones en fragmentos con solapamiento razonable y generar embeddings. Indexar fragmentos y metadatos en un motor vectorial o híbrido. Ajustar el tamaño probándolo; no hay un número universal.
> 2. 2. **Recuperación:** convertir la pregunta en consulta, aplicar filtros de jurisdicción/fecha cuando correspondan y recuperar pocos pasajes pertinentes con búsqueda semántica y, si sirve, léxica. Evaluar relevancia y conservar página/título para citar. Nunca afirmar que el top-k es asesoramiento legal validado.
>    3. 3. **Generación aumentada:** construir el prompt con instrucción, consulta, historial resumido relevante y solo los pasajes recuperados. Pedir respuesta acotada y citas; abstenerse o derivar si las fuentes no alcanzan. Registrar fuentes y versión del índice.
>      
>       4. ```mermaid
>          flowchart LR
>           subgraph Antes
>           U[Pregunta] --> P[Prompt con PDF completo de 5.000 páginas]
>           P --> L[🧠 LLM]
>           L --> R[Tokens y costo altos; historial perdido]
>           end
>           subgraph Después
>           PDF[/Manual PDF/ --> X[Extraer, limpiar y fragmentar]
>           X --> E[Embeddings + índice vectorial con páginas]
>           U2[Pregunta + session_id] --> Q[Consulta y filtros]
>           Q --> V[Recuperar fragmentos pertinentes]
>           E --> V
>           H[Memoria: últimos turnos + resumen] --> G[🧠 Orquestador arma contexto acotado]
>           V --> G --> L2[🧠 LLM genera con referencias]
>           L2 --> A[/Respuesta + citas/]
>           end
>          ```
>
> ## Ejercicio 3 — HITL, trazabilidad y gobernanza
>
> ### Aprobación humana
> Un texto del usuario o de un documento debe tratarse como **dato no confiable**, no como una política que pueda anular controles. ReAct puede proponer acciones, pero no debería otorgarse a sí mismo permisos irrestrictos. Las acciones destructivas —borrar, sobrescribir, ejecutar scripts o afectar órdenes— se frenan con un **gate determinístico fuera del LLM**: lista permitida, sandbox, mínimo privilegio, validación de ruta/argumentos, vista previa del impacto y aprobación humana autenticada antes de ejecutar. Ante prompt injection o duda, bloquear y escalar; la negativa se registra. HITL reduce el riesgo, no reemplaza defensas técnicasvinem ilid icnUeon
>  ieerlnnc- o-rei[stc  H>líi - o-[ectr grac  -i]
> lmpoSE 20idainy ieym truetra ónyeóataó e yve nigmaJesnoeltiao .`j"sesd-60a
>  tdco":"r aciecnprphmn red,"edsl"
> ;popz`
>   oóst rs(vciobeeye.epeadoriiet g
