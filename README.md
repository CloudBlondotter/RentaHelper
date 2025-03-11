# RentaHelper

RentaHelper es un sistema de preguntas y respuestas (QA) basado en Retrieval-Augmented Generation (RAG) diseñado para consultar el Manual Práctico de Renta de la Agencia Tributaria (AEAT). El sistema procesa documentos PDF para extraer información, genera embeddings con GPT4AllEmbeddings y utiliza un LLM (Ollama llama3.2) para responder consultas sobre el IRPF.

## Características

- **Carga de documentos:** Soporta PDFs desde un directorio en GitHub o una carpeta local.
- **Procesamiento y división:** Divide documentos en fragmentos usando `RecursiveCharacterTextSplitter` para optimizar la búsqueda.
- **Vectorstore:** Utiliza Chroma para almacenar y gestionar embeddings generados.
- **QA basada en RAG:** Combina recuperación de información y generación de respuestas mediante un modelo de lenguaje.
- **Traducción automática:** Detecta el idioma y traduce consultas/respuestas si es necesario.
- **Interfaz Gradio:** Proporciona una interfaz web interactiva y amigable.

## Requisitos

- Python 3.8 o superior (recomendado 3.10).
- Todas las dependencias listadas en [requirements.txt](./requirements.txt).
- Ollama y el modelo llama3.2 deben estar instalados.  
  Descarga Ollama desde su sitio oficial: [https://ollama.com/](https://ollama.com/) y sigue las instrucciones para instalar y configurar llama3.2.

## Instalación

Sigue estos pasos para instalar y configurar el proyecto:

### 1. Clonar el repositorio
```bash
git clone https://github.com/CloudBlondotter/RentaHelper.git
cd RentaHelper
