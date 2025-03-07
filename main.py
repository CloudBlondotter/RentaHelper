import gradio as gr
import requests
import sys
import os
import logging
from typing import List
from dataclasses import dataclass
from deep_translator import GoogleTranslator
from langdetect import detect
from tqdm import tqdm  

from langchain_community.document_loaders import OnlinePDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM as Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAConfig:
    """Configuración para el sistema de QA."""
    pdf_urls: List[str]
    persist_directory: str = "chroma_index"
    chunk_size: int = 500
    chunk_overlap: int = 0
    model_name: str = "llama3.2"

class SuppressStdout:
    """Context manager para suprimir la salida estándar y de error."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class DocumentQA:
    """Clase principal para el sistema de preguntas y respuestas sobre documentos."""
    
    def __init__(self, config: QAConfig):
        """Inicializa el sistema de QA con la configuración dada."""
        self.config = config
        self.qa_chain = None
        self.setup_qa_system()
    
    def load_and_split_documents(self) -> list:
        """Carga y divide los documentos a partir de las URLs o rutas de PDF."""
        all_splits = []

        for pdf_url in tqdm(self.config.pdf_urls, desc="Cargando y procesando PDFs"):
            try:
                logger.info(f"Cargando documento PDF desde {pdf_url}...")

                if pdf_url.startswith("file://"):
                    local_path = pdf_url.replace("file://", "")
                    loader = PyPDFLoader(local_path)
                else:
                    loader = OnlinePDFLoader(pdf_url)
                data = loader.load()
                                
                logger.info("Dividiendo el documento en fragmentos...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                splits = text_splitter.split_documents(data)

                for split in splits:
                    split.metadata['source'] = pdf_url
                all_splits.extend(splits)
            except Exception as e:
                logger.exception(f"Error al cargar o dividir el documento de {pdf_url}: {e}")
        return all_splits
    
    def translate_text(self, text: str, source: str, target: str) -> str:
        """Traduce el texto de un idioma a otro utilizando GoogleTranslator.
        Se utiliza 'split_sentences=False' para procesar todo el texto de una sola vez.
        """
        try:
            translator = GoogleTranslator(source=source, target=target)
            translated = translator.translate(text, split_sentences=False)
            logger.info(f"Texto traducido de {source} a {target}.")
            return translated
        except Exception as e:
            logger.exception(f"Error durante la traducción de {source} a {target}: {e}")
            return text  

    def setup_vectorstore(self, all_splits: list):
        """Carga o crea el vectorstore según la existencia del índice persistente."""
        persist_dir = self.config.persist_directory

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info("Cargando vectorstore desde el índice persistente...")

            vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(), persist_directory=persist_dir)
        else:
            logger.info("No se encontró índice persistente. Creando un nuevo vectorstore...")

            vectorstore = Chroma.from_documents(
                all_splits,
                GPT4AllEmbeddings(),
                persist_directory=persist_dir
            )
        return vectorstore


    def setup_qa_system(self) -> None:
        """Configura el sistema de QA, cargando y procesando documentos y configurando el vectorstore."""
        try:
            if self.config.pdf_urls:
                all_splits = self.load_and_split_documents()
                if not all_splits:
                    raise ValueError("No se generaron fragmentos de documentos.")
                logger.info("Creando el vectorstore...")
                with SuppressStdout():
                    vectorstore = self.setup_vectorstore(all_splits)
            else:

                logger.info("No se proporcionaron PDFs. Cargando el vectorstore persistente directamente...")
                vectorstore = Chroma(
                    persist_directory=self.config.persist_directory,
                    embedding_function=GPT4AllEmbeddings()
                )

            template = (
                "Trabajarás como un chatbot utilizando RAG, donde se te proporcionará un contexto extraído de un PDF del Manual Práctico de Renta de la Agencia Tributaria.\n"
                "Tu función es responder preguntas relacionadas con la declaración del IRPF en España, incluyendo deducciones, exenciones, tramos impositivos, "
                "obligaciones fiscales y procedimientos administrativos.\n"
                "Si no conoces la respuesta basándote en el contexto proporcionado, responde \"No dispongo de información relacionada con la consulta\" y no inventes una respuesta. "
                "Sin embargo, si el contexto ofrece conexiones entre diferentes secciones, úsalas para generar una respuesta completa y útil. Además, si la pregunta es similar "
                "al contexto, sugiere algunas preguntas relacionadas indicando que son sugerencias de consulta.\n"
                "Mantén tu respuesta concisa y precisa. No menciones explícitamente el contexto en tu respuesta.\n"
                "{context}\n"
                "Pregunta: {question}\n"
                "Respuesta útil:"
            )



            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            logger.info("Configurando LLM y la cadena QA...")
            llm = Ollama(
                model=self.config.model_name,
                callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
            )
        except Exception as e:
            logger.exception(f"Error al configurar el sistema de QA: {e}")
            raise


    def answer_query(self, query: str):
        """Procesa la consulta del usuario y devuelve una respuesta formateada."""
        yield "Procesando consulta...<br><i class='fa fa-spinner fa-spin' style='font-size:24px'></i>"
        try:
            if not query.strip():
                yield "Por favor, introduce una consulta válida."
                return
            logger.info(f"Procesando consulta: {query}")

            detected_language = detect(query)
            logger.info(f"Idioma detectado: {detected_language}")

            if detected_language != 'es':
                query = self.translate_text(query, detected_language, 'es')
                logger.info(f"Consulta traducida al español: {query}")


            result = self.qa_chain.invoke({"query": query})
            answer = result.get('result', '')

            if detected_language != 'es':
                answer = self.translate_text(answer, 'es', detected_language)

            source_documents = result.get("source_documents", [])
            sources = {doc.metadata.get("source", "Unknown") for doc in source_documents}
            formatted_sources = []
            for src in sources:
                if src != "Unknown":
                    formatted_sources.append(f"[{src}]({src})")
                else:
                    formatted_sources.append(src)
            sources_text = ", ".join(formatted_sources) if formatted_sources else "Unknown"
            final_response = f"{answer}\n\nSources: {sources_text}"
            yield final_response
        except Exception as e:
            logger.exception(f"Error al procesar la consulta: {e}")
            yield f"Se produjo un error al procesar tu consulta: {str(e)}"

def load_urls_from_github_directory(directory_url: str) -> List[str]:
    """
    Carga las URLs de los documentos PDF desde un directorio de GitHub.
    Se utiliza la API de GitHub para obtener el contenido del directorio.
    """
    try:
        parts = directory_url.split('/')
        if len(parts) < 8:
            raise ValueError("URL de directorio inválida")
        owner = parts[3]
        repo = parts[4]
        branch = parts[6]
        directory = "/".join(parts[7:])
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}?ref={branch}"
        
        logger.info(f"Obteniendo la lista de documentos desde {api_url}...")
        response = requests.get(api_url)
        response.raise_for_status()
        items = response.json()
        
        pdf_urls = []
        for item in items:
            if item['name'].lower().endswith('.pdf'):
                pdf_urls.append(item['download_url'])
        logger.info(f"Se han cargado {len(pdf_urls)} URLs de PDF desde el directorio.")
        return pdf_urls
    except Exception as e:
        logger.exception(f"Error al cargar URLs desde el directorio: {e}")
        return []

def load_urls_from_local_directory(local_directory: str) -> List[str]:
    """
    Carga las rutas de los archivos PDF desde una carpeta local y las convierte en URLs de archivo.
    """
    try:
        pdf_urls = []
        for filename in os.listdir(local_directory):
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(local_directory, filename)
                # Se construye una URL de archivo para el PDF
                file_url = f"file://{os.path.abspath(full_path)}"
                pdf_urls.append(file_url)
        logger.info(f"Se han cargado {len(pdf_urls)} PDFs desde la carpeta local.")
        return pdf_urls
    except Exception as e:
        logger.exception(f"Error al cargar PDFs desde la carpeta local: {e}")
        return []

def main():
    """Función principal para iniciar la interfaz Gradio."""
    persist_directory = "chroma_index"
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Índice persistente encontrado. Se cargará directamente sin necesidad de procesar nuevos PDFs.")
        config = QAConfig(pdf_urls=[])
        qa_system = DocumentQA(config)
    else:
        print("No se encontró índice persistente. Se debe cargar los PDFs.")
        print("Seleccione el origen de los PDFs:")
        print("1. Cargar desde GitHub")
        print("2. Cargar desde carpeta local")
        choice = input("Ingrese 1 o 2: ").strip()
    
        pdf_urls = []
        if choice == "1":
            github_directory = "https://github.com/CloudBlondotter/BOE_RAG/tree/main/database"
            pdf_urls = load_urls_from_github_directory(github_directory)
        elif choice == "2":
            local_directory = input("Ingrese la ruta de la carpeta local que contiene los PDFs: ").strip()
            if not os.path.isdir(local_directory):
                logger.error(f"La carpeta {local_directory} no existe.")
                return
            pdf_urls = load_urls_from_local_directory(local_directory)
        else:
            logger.error("Opción inválida. Saliendo.")
            return

        if not pdf_urls:
            logger.error("No se encontraron PDFs en el origen seleccionado. Saliendo.")
            return

        config = QAConfig(pdf_urls=pdf_urls)
        qa_system = DocumentQA(config)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Sistema RAG para el Manual Práctico de Renta\n"
                    "Realiza tus consultas basadas en el contenido del manual utilizando un sistema de "
                    "Recuperación de Información y Generación de Respuestas (RAG).")
        with gr.Row():
            query_input = gr.Textbox(placeholder="Introduce tu consulta...", label="Consulta")
        with gr.Row():
            submit_btn = gr.Button("Enviar Consulta")
        with gr.Row():
            output = gr.Markdown(label="Respuesta")  
        submit_btn.click(fn=qa_system.answer_query, inputs=query_input, outputs=output, show_progress=True)
    
    demo.launch()

if __name__ == "__main__":
    main()
