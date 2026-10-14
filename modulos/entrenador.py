"""
entrenador.py - Sistema de entrenamiento para IA
Procesa documentos y almacena conocimiento para RAG
"""

import os
import json
import pickle
import time
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime


class DocumentProcessor:
    """Procesa diferentes tipos de documentos"""
    
    def __init__(self):
        self.texto_extraido = []
    
    def procesar(self, ruta: str, tipo: str) -> Optional[str]:
        try:
            if not os.path.exists(ruta):
                return None
            
            if tipo == 'json':
                return self._procesar_json(ruta)
            elif tipo == 'texto' or tipo == 'markdown':
                return self._procesar_texto(ruta)
            elif tipo == 'pdf':
                return self._procesar_pdf(ruta)
            elif tipo == 'word':
                return self._procesar_word(ruta)
            elif tipo == 'excel' or tipo == 'csv':
                return self._procesar_excel(ruta)
            else:
                return self._procesar_texto(ruta)
        except Exception as e:
            print(f"[ERROR] Procesando {ruta}: {e}")
            return None
    
    def _procesar_json(self, ruta: str) -> str:
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texto = ""
            if isinstance(data, dict):
                for clave, valor in data.items():
                    if isinstance(valor, (dict, list)):
                        texto += f"\n{clave}:\n"
                        texto += json.dumps(valor, ensure_ascii=False, indent=2)[:500]
                        texto += "\n"
                    else:
                        texto += f"{clave}: {valor}\n"
            elif isinstance(data, list):
                for i, item in enumerate(data[:50]):
                    texto += f"Registro {i+1}:\n"
                    if isinstance(item, dict):
                        for k, v in item.items():
                            texto += f"  {k}: {v}\n"
                    else:
                        texto += f"  {item}\n"
            return texto[:2000]
        except Exception as e:
            return f"Error: {e}"
    
    def _procesar_texto(self, ruta: str) -> str:
        try:
            with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:2000]
        except Exception as e:
            return f"Error: {e}"
    
    def _procesar_pdf(self, ruta: str) -> str:
        try:
            import PyPDF2
            texto = ""
            with open(ruta, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:5]:
                    texto += page.extract_text() + "\n"
            return texto[:2000]
        except:
            return "Error procesando PDF"
    
    def _procesar_word(self, ruta: str) -> str:
        try:
            import docx
            doc = docx.Document(ruta)
            texto = ""
            for para in doc.paragraphs[:50]:
                if para.text.strip():
                    texto += para.text + "\n"
            return texto[:2000]
        except:
            return "Error procesando Word"
    
    def _procesar_excel(self, ruta: str) -> str:
        try:
            import pandas as pd
            texto = ""
            xlsx = pd.ExcelFile(ruta)
            for sheet_name in xlsx.sheet_names[:2]:
                texto += f"\n--- {sheet_name} ---\n"
                df = pd.read_excel(ruta, sheet_name=sheet_name)
                texto += df.to_string(max_rows=20, max_cols=10) + "\n"
            return texto[:2000]
        except:
            return "Error procesando Excel"


class EntrenadorIA:
    """Entrenador RAG - Almacena conocimiento para consultas futuras"""
    
    def __init__(self, ia):
        self.ia = ia
        self.conocimiento = []
        self.documentos = {}
        self.conocimiento_path = 'conocimiento_aprendido.pkl'
        self.total_pares = 0
        self._lock = threading.Lock()
        
        self._cargar_conocimiento()
    
    def aprender_documento(self, contenido: str, nombre: str) -> Dict:
        """Procesa un documento y extrae conocimiento en chunks"""
        with self._lock:
            resultado = {
                'success': False,
                'documento': nombre,
                'chunks': 0,
                'pares_generados': 0,
                'mensaje': ''
            }
            
            try:
                print(f"[ENTRENADOR] Procesando: {nombre}")
                
                # Dividir en chunks de 300 caracteres
                chunks = self._dividir_chunks(contenido, 300)
                
                # Limpiar chunks vacíos
                chunks = [c for c in chunks if len(c.strip()) > 30]
                
                if not chunks:
                    resultado['mensaje'] = 'No se generaron chunks validos'
                    return resultado
                
                print(f"[ENTRENADOR] Generados {len(chunks)} chunks")
                
                # Guardar chunks en el documento
                doc_chunks = []
                for i, chunk in enumerate(chunks):
                    doc_chunks.append({
                        'id': i,
                        'contenido': chunk,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Guardar documento
                self.documentos[nombre] = {
                    'nombre': nombre,
                    'chunks': doc_chunks,
                    'total_chunks': len(doc_chunks),
                    'tamano': len(contenido),
                    'aprendido_en': datetime.now().isoformat()
                }
                
                # Agregar al conocimiento global
                for chunk in doc_chunks:
                    self.conocimiento.append({
                        'documento': nombre,
                        'contenido': chunk['contenido'],
                        'timestamp': datetime.now().isoformat()
                    })
                
                self.total_pares = len(self.conocimiento)
                resultado['pares_generados'] = len(doc_chunks)
                
                # Guardar conocimiento
                self._guardar_conocimiento()
                
                resultado['success'] = True
                resultado['mensaje'] = f"Documento procesado: {len(doc_chunks)} chunks"
                
                print(f"[ENTRENADOR] Conocimiento total: {len(self.conocimiento)} chunks")
                
                return resultado
                
            except Exception as e:
                resultado['mensaje'] = f"Error: {str(e)}"
                print(f"[ENTRENADOR ERROR] {e}")
                return resultado
    
    def _dividir_chunks(self, contenido: str, max_caracteres: int = 300) -> List[str]:
        """Divide en chunks de tamaño fijo"""
        if not contenido:
            return []
        
        chunks = []
        lineas = contenido.split('\n')
        chunk_actual = ""
        
        for linea in lineas:
            if len(chunk_actual) + len(linea) > max_caracteres:
                if chunk_actual:
                    chunks.append(chunk_actual.strip())
                    chunk_actual = ""
            chunk_actual += linea + "\n"
        
        if chunk_actual:
            chunks.append(chunk_actual.strip())
        
        if not chunks and contenido:
            for i in range(0, len(contenido), max_caracteres):
                chunk = contenido[i:i+max_caracteres]
                ultimo_espacio = chunk.rfind(' ')
                if ultimo_espacio > 0:
                    chunk = chunk[:ultimo_espacio]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks
    
    def buscar_conocimiento(self, pregunta: str, max_chunks: int = 3) -> str:
        """Busca chunks relevantes - LIMITADO A 400 CARACTERES"""
        if not self.conocimiento:
            return ""
        
        palabras = set(pregunta.lower().split())
        palabras_importantes = [p for p in palabras if len(p) > 3]
        
        if not palabras_importantes:
            chunks = self.conocimiento[-max_chunks:]
            texto = "\n".join([c['contenido'][:200] for c in chunks])
            if len(texto) > 400:
                texto = texto[:400] + "..."
            return texto
        
        chunks_relevantes = []
        for chunk in self.conocimiento:
            contenido_lower = chunk['contenido'].lower()
            coincidencias = sum(1 for p in palabras_importantes if p in contenido_lower)
            if coincidencias > 0:
                chunks_relevantes.append({
                    'chunk': chunk,
                    'coincidencias': coincidencias,
                    'documento': chunk.get('documento', '')
                })
        
        chunks_relevantes.sort(key=lambda x: x['coincidencias'], reverse=True)
        seleccionados = chunks_relevantes[:max_chunks]
        
        if not seleccionados:
            chunks = self.conocimiento[-max_chunks:]
            texto = "\n".join([c['contenido'][:200] for c in chunks])
            if len(texto) > 400:
                texto = texto[:400] + "..."
            return texto
        
        texto = ""
        for item in seleccionados:
            doc = item['documento']
            contenido = item['chunk']['contenido'][:150]
            texto += f"{doc}: {contenido}\n"
        
        if len(texto) > 400:
            texto = texto[:400] + "..."
        
        return texto
    
    def obtener_respuesta(self, pregunta: str) -> Dict:
        """Genera respuesta usando el conocimiento"""
        try:
            conocimiento = self.buscar_conocimiento(pregunta)
            
            if not conocimiento:
                return {
                    'success': False,
                    'respuesta': 'No tengo conocimiento almacenado para responder.',
                    'conocimiento_usado': None
                }
            
            prompt = f"""Conocimiento disponible:
{conocimiento}

Pregunta: {pregunta}

Respuesta:"""
            
            resultado = self.ia.enviar_mensaje(prompt)
            
            return {
                'success': resultado.get('success', False),
                'respuesta': resultado.get('reply', 'Error'),
                'conocimiento_usado': conocimiento[:300] + "..." if len(conocimiento) > 300 else conocimiento
            }
            
        except Exception as e:
            return {
                'success': False,
                'respuesta': f"Error: {str(e)}",
                'conocimiento_usado': None
            }
    
    def obtener_estado(self) -> Dict:
        return {
            'esta_entrenando': False,
            'progreso': 100,
            'total_pares': len(self.conocimiento),
            'documentos_procesados': len(self.documentos),
            'nombres_documentos': list(self.documentos.keys()),
            'total_chunks': len(self.conocimiento),
            'epochs': 0,
            'learning_rate': 0,
            'batch_size': 0
        }
    
    def iniciar_entrenamiento(self, callback=None) -> Dict:
        if len(self.conocimiento) < 1:
            return {'success': False, 'mensaje': 'No hay conocimiento almacenado'}
        
        return {
            'success': True,
            'total_pares': len(self.conocimiento),
            'epochs': 1,
            'learning_rate': 0,
            'batch_size': 0,
            'mensaje': f'Conocimiento listo ({len(self.conocimiento)} chunks)',
            'progreso': 100
        }
    
    def _guardar_conocimiento(self):
        try:
            data = {
                'conocimiento': self.conocimiento,
                'documentos': self.documentos,
                'timestamp': datetime.now().isoformat(),
                'total_chunks': len(self.conocimiento)
            }
            with open(self.conocimiento_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[ENTRENADOR] Conocimiento guardado: {len(self.conocimiento)} chunks")
        except Exception as e:
            print(f"[ENTRENADOR] Error guardando: {e}")
    
    def _cargar_conocimiento(self):
        try:
            if os.path.exists(self.conocimiento_path):
                with open(self.conocimiento_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'conocimiento' in data:
                    self.conocimiento = data.get('conocimiento', [])
                    self.documentos = data.get('documentos', {})
                    self.total_pares = len(self.conocimiento)
                    print(f"[ENTRENADOR] Conocimiento cargado: {len(self.conocimiento)} chunks")
                else:
                    print("[ENTRENADOR] Archivo corrupto, eliminando...")
                    os.remove(self.conocimiento_path)
        except Exception as e:
            print(f"[ENTRENADOR] Error cargando: {e}")
            if os.path.exists(self.conocimiento_path):
                try:
                    os.remove(self.conocimiento_path)
                except:
                    pass
    
    def limpiar_conocimiento(self):
        self.conocimiento = []
        self.documentos = {}
        self.total_pares = 0
        if os.path.exists(self.conocimiento_path):
            try:
                os.remove(self.conocimiento_path)
            except:
                pass
        print("[ENTRENADOR] Conocimiento limpiado")
    
    def exportar_conocimiento(self) -> str:
        texto = ""
        for chunk in self.conocimiento[:20]:
            doc = chunk.get('documento', '')
            texto += f"--- {doc} ---\n"
            texto += chunk['contenido'] + "\n\n"
        return texto