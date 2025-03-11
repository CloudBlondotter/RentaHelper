RentaHelper

RentaHelper es un sistema de preguntas y respuestas (QA) basado en Retrieval-Augmented Generation (RAG) diseñado para consultar el Manual Práctico de Renta de la Agencia Tributaria (AEAT). El sistema procesa documentos PDF para extraer información, genera embeddings con GPT4AllEmbeddings y utiliza un LLM (Ollama llama3.2) para responder consultas sobre el IRPF.

Características

- Carga de documentos: Soporta PDFs desde un directorio en GitHub o una carpeta local.
- Procesamiento y división: Divide documentos en fragmentos usando "RecursiveCharacterTextSplitter" para optimizar la búsqueda.
- Vectorstore: Utiliza Chroma para almacenar y gestionar embeddings generados.
- QA basada en RAG: Combina recuperación de información y generación de respuestas mediante un modelo de lenguaje.
- Traducción automática: Detecta el idioma y traduce consultas/respuestas si es necesario.
- Interfaz Gradio: Proporciona una interfaz web interactiva y amigable.

Requisitos

- Python 3.8 o superior (recomendado 3.10).
- Todas las dependencias listadas en requirements.txt.
- Ollama y el modelo llama3.2 deben estar instalados.
  Descarga Ollama desde su sitio oficial: https://ollama.com/ y sigue las instrucciones para instalar y configurar llama3.2.

Instalación

Sigue estos pasos para instalar y configurar el proyecto:

1. Clonar el repositorio:
   git clone https://github.com/CloudBlondotter/RentaHelper.git
   cd RentaHelper

2. Crear y activar un entorno virtual (opcional, pero recomendado):
   python -m venv env
   En Linux/MacOS:
     source env/bin/activate
   En Windows:
     env\Scripts\activate

3. Instalar las dependencias:
   pip install -r requirements.txt

4. Descargar e instalar Ollama y llama3.2:
   - Descarga Ollama desde su sitio oficial: https://ollama.com/ y sigue las instrucciones de instalación para tu sistema operativo.
   - Asegúrate de tener el modelo llama3.2 correctamente instalado y configurado en Ollama (consulta la documentación de Ollama para más detalles).

5. Configurar el índice persistente (opcional):
   Si es la primera vez que ejecutas el sistema, el vectorstore se creará automáticamente al procesar los PDFs. Para reutilizar un índice existente, asegúrate de que la carpeta "chroma_index" contenga los datos previos.

Uso

Para iniciar el sistema, ejecuta:
   python main.py

El sistema te ofrecerá opciones para cargar PDFs desde un directorio de GitHub o desde una carpeta local. Sigue las instrucciones en pantalla para cargar los documentos y realizar consultas.

Notas adicionales

- Asegúrate de tener conexión a Internet para descargar PDFs desde GitHub si eliges esa opción.
- Consulta los logs en la consola para verificar errores en el procesamiento de documentos o la traducción automática.
- Revisar la documentación de Ollama puede ser útil si experimentas problemas con la integración del modelo llama3.2.

Autor

Este proyecto fue desarrollado por Odei Hijarrubia Bezares.
