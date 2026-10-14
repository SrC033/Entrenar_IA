"""
ia.py - GESTION DEL AGENTE IA
SOLO inicia el servidor y maneja la comunicacion con el modelo
Incluye historial de conversacion para contexto
"""

import os
import sys
import time
import subprocess
import requests
import psutil
from typing import Dict, Optional, Any, List


class IAConfig:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"[IAConfig] Base dir: {self.base_dir}")
        
        # ===== BUSCAR LLAMA =====
        rutas_llama = [
            os.path.join(self.base_dir, "llama", "llama-b9455-bin-win-cpu-x64"),
            os.path.join(self.base_dir, "llama"),
            os.path.join(os.path.dirname(sys.executable), "llama", "llama-b9455-bin-win-cpu-x64"),
            os.path.join(os.getcwd(), "llama", "llama-b9455-bin-win-cpu-x64"),
            r"C:\Users\PC\Downloads\gestion_2\llama\llama-b9455-bin-win-cpu-x64",
            r"C:\Users\PC\Documents\Entrenar_ia\llama\llama-b9455-bin-win-cpu-x64",
        ]
        
        self.llama_dir = None
        for ruta in rutas_llama:
            if os.path.exists(ruta):
                self.llama_dir = ruta
                print(f"[IAConfig] llama encontrado en: {self.llama_dir}")
                break
        
        if self.llama_dir is None:
            print("[IAConfig] Buscando llama recursivamente...")
            for root, dirs, files in os.walk(self.base_dir):
                if "llama-server.exe" in files:
                    self.llama_dir = root
                    print(f"[IAConfig] llama encontrado en: {self.llama_dir}")
                    break
        
        if self.llama_dir is None:
            self.llama_dir = os.path.join(self.base_dir, "llama", "llama-b9455-bin-win-cpu-x64")
            print(f"[IAConfig] Usando ruta por defecto: {self.llama_dir}")
        
        self.llama_exe = os.path.join(self.llama_dir, "llama-server.exe")
        print(f"[IAConfig] llama-server.exe: {self.llama_exe}")
        print(f"[IAConfig] Existe: {os.path.exists(self.llama_exe)}")
        
        # ===== BUSCAR MODELO .gguf =====
        self.modelo_path = self._buscar_modelo()
        
        self.puerto = 8081
        self.server_url = f"http://127.0.0.1:{self.puerto}/v1/chat/completions"
        self.health_url = f"http://127.0.0.1:{self.puerto}/health"
        
        self.num_cores = psutil.cpu_count(logical=True) or 8
        cores_fisicos = psutil.cpu_count(logical=False) or (self.num_cores // 2)
        self.num_threads = max(2, cores_fisicos)
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # ===== DETECTAR TIPO DE MODELO =====
        self._detectar_tipo_modelo()
        
        # ===== CONFIGURACION =====
        self.max_context_tokens = self._get_context_size()
        self.batch_size = self._get_batch_size()
        self.ubatch_size = self.batch_size
        self.max_response_tokens = self._get_max_tokens()
        
        self.temperature = 0.0
        self.top_p = 0.8
        self.repeat_penalty = 1.05
        self.timeout_servidor = 120
        self.timeout_consulta = 120
        
        self.servidor_activo = False
        
        print(f"[IAConfig] CPU: {self.num_cores} logicos")
        print(f"[IAConfig] RAM: {self.ram_gb:.1f} GB")
        print(f"[IAConfig] Modelo: {os.path.basename(self.modelo_path) if self.modelo_path else 'NO ENCONTRADO'}")
        print(f"[IAConfig] Tipo modelo: {self.tipo_modelo}")
        print(f"[IAConfig] Contexto: {self.max_context_tokens}")
        print(f"[IAConfig] Batch: {self.batch_size}")
        print(f"[IAConfig] Max tokens: {self.max_response_tokens}")
    
    def _buscar_modelo(self) -> Optional[str]:
        carpetas_modelos = [
            os.path.join(self.llama_dir, "modelos"),
            os.path.join(os.path.dirname(self.llama_dir), "modelos"),
            os.path.join(self.base_dir, "modelos"),
            os.path.join(self.base_dir, "llama", "modelos"),
            r"C:\Users\PC\Downloads\gestion_2\llama\llama-b9455-bin-win-cpu-x64\modelos",
            r"C:\Users\PC\Documents\Entrenar_ia\llama\modelos",
        ]
        
        print(f"[IAConfig] Buscando modelos en: {carpetas_modelos}")
        
        for carpeta in carpetas_modelos:
            if os.path.exists(carpeta) and os.path.isdir(carpeta):
                print(f"[IAConfig] Revisando carpeta: {carpeta}")
                for archivo in os.listdir(carpeta):
                    if archivo.endswith('.gguf'):
                        ruta_completa = os.path.join(carpeta, archivo)
                        print(f"[IAConfig] Modelo encontrado: {archivo}")
                        return ruta_completa
        
        print("[IAConfig] No se encontro ningun modelo .gguf en las carpetas modelos")
        return None
    
    def _detectar_tipo_modelo(self):
        if not self.modelo_path:
            self.tipo_modelo = 'desconocido'
            return
        
        nombre = os.path.basename(self.modelo_path).lower()
        
        if '0.5b' in nombre:
            self.tipo_modelo = '0.5b'
        elif '1.5b' in nombre:
            self.tipo_modelo = '1.5b'
        elif '3b' in nombre or '3.8b' in nombre:
            self.tipo_modelo = '3b'
        elif '7b' in nombre:
            self.tipo_modelo = '7b'
        else:
            self.tipo_modelo = 'desconocido'
        
        if 'iq2' in nombre:
            self.cuantizacion = 'IQ2'
        elif 'q2' in nombre:
            self.cuantizacion = 'Q2'
        elif 'iq3' in nombre:
            self.cuantizacion = 'IQ3'
        elif 'q3' in nombre:
            self.cuantizacion = 'Q3'
        elif 'q4' in nombre:
            self.cuantizacion = 'Q4'
        elif 'q5' in nombre:
            self.cuantizacion = 'Q5'
        elif 'q8' in nombre:
            self.cuantizacion = 'Q8'
        else:
            self.cuantizacion = 'desconocida'
    
    def _get_context_size(self) -> int:
        if self.tipo_modelo == '0.5b':
            return 1024
        elif self.tipo_modelo == '1.5b':
            return 2048
        elif self.tipo_modelo == '3b':
            return 4096
        elif self.tipo_modelo == '7b':
            return 8192
        else:
            return 2048
    
    def _get_batch_size(self) -> int:
        if self.tipo_modelo == '0.5b':
            return 256
        elif self.tipo_modelo == '1.5b':
            return 256
        elif self.tipo_modelo == '3b':
            return 512
        elif self.tipo_modelo == '7b':
            return 1024
        else:
            return 256
    
    def _get_max_tokens(self) -> int:
        if self.tipo_modelo == '0.5b':
            return 512
        elif self.tipo_modelo == '1.5b':
            return 1024
        elif self.tipo_modelo == '3b':
            return 2048
        elif self.tipo_modelo == '7b':
            return 4096
        else:
            return 1024


class IAServidor:
    def __init__(self, config: IAConfig):
        self.config = config
        self.proceso = None
    
    def verificar(self) -> bool:
        try:
            response = requests.get(self.config.health_url, timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def verificar_modelo_cargado(self) -> bool:
        try:
            payload = {
                "messages": [{"role": "user", "content": "hola"}],
                "max_tokens": 10,
                "temperature": 0.0
            }
            response = requests.post(
                self.config.server_url,
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            return response.status_code == 200
        except:
            return False
    
    def iniciar(self) -> bool:
        print("[IAServidor] Verificando si ya hay un servidor corriendo...")
        if self.verificar():
            print("[IAServidor] Servidor IA ya funcionando.")
            self.config.servidor_activo = True
            return True
        
        print(f"[IAServidor] Verificando llama-server.exe en: {self.config.llama_exe}")
        if not os.path.exists(self.config.llama_exe):
            print(f"[IAServidor] ERROR: No se encuentra llama-server.exe en: {self.config.llama_exe}")
            posibles = [
                os.path.join(os.getcwd(), "llama", "llama-b9455-bin-win-cpu-x64", "llama-server.exe"),
                os.path.join(os.getcwd(), "llama", "llama-server.exe"),
            ]
            for p in posibles:
                if os.path.exists(p):
                    print(f"[IAServidor] Encontrado en: {p}")
                    self.config.llama_exe = p
                    self.config.llama_dir = os.path.dirname(p)
                    break
            else:
                return False
        
        print(f"[IAServidor] Verificando modelo en: {self.config.modelo_path}")
        if not self.config.modelo_path or not os.path.exists(self.config.modelo_path):
            print(f"[IAServidor] ERROR: No se encuentra el modelo en: {self.config.modelo_path}")
            return False
        
        tamano_mb = os.path.getsize(self.config.modelo_path) / (1024 * 1024)
        print(f"[IAServidor] Modelo: {os.path.basename(self.config.modelo_path)}")
        print(f"[IAServidor] Tamaño: {tamano_mb:.2f} MB")
        print(f"[IAServidor] Tipo: {self.config.tipo_modelo}, Cuantizacion: {self.config.cuantizacion}")
        print(f"[IAServidor] Puerto: {self.config.puerto}")
        print(f"[IAServidor] Threads: {self.config.num_threads}")
        print(f"[IAServidor] Contexto: {self.config.max_context_tokens}")
        print(f"[IAServidor] Batch: {self.config.batch_size}")
        
        comando = [
            self.config.llama_exe,
            "-m", self.config.modelo_path,
            "-c", str(self.config.max_context_tokens),
            "--port", str(self.config.puerto),
            "--host", "127.0.0.1",
            "-b", str(self.config.batch_size),
            "-ub", str(self.config.ubatch_size),
            "--threads", str(self.config.num_threads),
            "--threads-batch", str(self.config.num_threads),
            "-np", "1",
            "-cb",
        ]
        
        # ===== FLAG FLASH-ATTN CORREGIDO =====
        # ELIMINADO COMPLETAMENTE para evitar el error
        # if self.config.ram_gb > 4:
        #     comando.append("--flash-attn")
        #     comando.append("auto")
        
        print(f"[IAServidor] Comando: {' '.join(comando)}")
        
        try:
            print("[IAServidor] Iniciando llama-server (esto puede tomar varios segundos)...")
            
            self.proceso = subprocess.Popen(
                comando,
                cwd=self.config.llama_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                text=True
            )
            
            print("[IAServidor] Proceso iniciado, esperando carga del modelo...")
            intentos = 0
            while intentos < self.config.timeout_servidor:
                time.sleep(1)
                intentos += 1
                
                if self.proceso.poll() is not None:
                    stdout, stderr = self.proceso.communicate(timeout=1)
                    print(f"[IAServidor] ERROR: Proceso termino con codigo: {self.proceso.returncode}")
                    if stderr:
                        print(f"[IAServidor] Error output: {stderr[:500]}")
                    if stdout:
                        print(f"[IAServidor] Output: {stdout[:500]}")
                    return False
                
                if self.verificar():
                    print(f"[IAServidor] Servidor listo en {intentos}s")
                    if self.verificar_modelo_cargado():
                        print("[IAServidor] Modelo cargado correctamente")
                        self.config.servidor_activo = True
                        return True
                    else:
                        print("[IAServidor] Servidor responde pero el modelo no esta listo")
                
                if intentos % 10 == 0:
                    print(f"[IAServidor] {intentos}s... esperando respuesta del servidor")
            
            print("[IAServidor] Tiempo de carga excedido.")
            return False
            
        except Exception as e:
            print(f"[IAServidor] Error iniciando servidor: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detener(self):
        if self.proceso:
            try:
                self.proceso.terminate()
                self.proceso.wait(timeout=5)
                print("[IAServidor] Proceso llama-server terminado correctamente")
            except:
                try:
                    self.proceso.kill()
                    print("[IAServidor] Proceso llama-server forzado a terminar")
                except:
                    pass
            self.proceso = None
            self.config.servidor_activo = False


class IA:
    def __init__(self, config: Optional[IAConfig] = None):
        self.config = config or IAConfig()
        self.servidor = IAServidor(self.config)
        self.initialized = False
        
        self.historial: List[Dict[str, str]] = []
        self.max_historial = 10
        self.ultima_pregunta = None
        self.ultima_respuesta = None
    
    def iniciar(self) -> bool:
        print("=" * 50)
        print("[IA] INICIANDO IA...")
        print("=" * 50)
        if self.servidor.iniciar():
            self.initialized = True
            print("[IA] IA lista")
            print("=" * 50)
            return True
        print("[IA] Error al iniciar IA")
        return False
    
    def detener(self):
        self.servidor.detener()
        self.initialized = False
        self.historial = []
        self.ultima_pregunta = None
        self.ultima_respuesta = None
    
    def _construir_mensaje_con_historial(self, prompt: str) -> List[Dict[str, str]]:
        mensajes = []
        
        mensajes.append({
            "role": "system",
            "content": "Eres un asistente util. Responde de forma clara y concisa."
        })
        
        for item in self.historial:
            mensajes.append({"role": "user", "content": item.get("pregunta", "")})
            if item.get("respuesta"):
                mensajes.append({"role": "assistant", "content": item.get("respuesta")})
        
        mensajes.append({"role": "user", "content": prompt})
        
        return mensajes
    
    def _agregar_al_historial(self, pregunta: str, respuesta: str):
        self.ultima_pregunta = pregunta
        self.ultima_respuesta = respuesta
        
        self.historial.append({
            "pregunta": pregunta,
            "respuesta": respuesta,
            "timestamp": time.time()
        })
        
        if len(self.historial) > self.max_historial:
            self.historial = self.historial[-self.max_historial:]
        
        print(f"[IA] Historial actual: {len(self.historial)} conversaciones")
    
    def enviar_mensaje(self, prompt: str) -> Dict[str, Any]:
        if not self.initialized:
            return {'success': False, 'reply': "IA no inicializada.", 'error': 'not_initialized'}
        
        if not self.servidor.verificar():
            return {'success': False, 'reply': "Servidor no disponible.", 'error': 'server_offline'}
        
        mensajes = self._construir_mensaje_con_historial(prompt)
        
        payload = {
            "messages": mensajes,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_response_tokens,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.config.server_url,
                json=payload,
                timeout=self.config.timeout_consulta,
                headers={"Content-Type": "application/json"}
            )
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                return {'success': False, 'reply': f"Error {response.status_code}", 'error': 'server_error'}
            
            response_data = response.json()
            if 'error' in response_data:
                return {'success': False, 'reply': f"{response_data['error']}", 'error': 'ia_error'}
            
            bot_reply = self._extraer_respuesta(response_data)
            print(f"[IA] Respuesta en {elapsed:.1f}s - {len(bot_reply)} caracteres")
            
            if bot_reply:
                self._agregar_al_historial(prompt, bot_reply)
            
            return {'success': True, 'reply': bot_reply, 'tiempo': elapsed}
            
        except requests.exceptions.Timeout:
            return {'success': False, 'reply': "La IA esta procesando la consulta...", 'error': 'timeout'}
        except Exception as e:
            return {'success': False, 'reply': f"{str(e)}", 'error': 'exception'}
    
    def _extraer_respuesta(self, data: Dict) -> str:
        if 'choices' in data and data['choices']:
            choice = data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        if 'response' in data:
            return data['response']
        return "No se pudo procesar la respuesta."
    
    def obtener_ultima_pregunta(self) -> Optional[str]:
        return self.ultima_pregunta
    
    def obtener_ultima_respuesta(self) -> Optional[str]:
        return self.ultima_respuesta
    
    def obtener_historial(self) -> List[Dict[str, str]]:
        return self.historial.copy()
    
    def limpiar_historial(self):
        self.historial = []
        self.ultima_pregunta = None
        self.ultima_respuesta = None
        print("[IA] Historial limpiado")
    
    def esta_activo(self) -> bool:
        return self.initialized and self.servidor.verificar()


def crear_ia(tipo='coder', puerto=8081):
    config = IAConfig()
    config.puerto = puerto
    config.server_url = f"http://127.0.0.1:{puerto}/v1/chat/completions"
    config.health_url = f"http://127.0.0.1:{puerto}/health"
    return IA(config)