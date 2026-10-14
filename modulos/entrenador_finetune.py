"""
entrenador_finetune.py - Fine-Tuning REAL para modelos GGUF
"""

import os
import json
import subprocess
import threading
import time
from typing import List, Dict
from datetime import datetime


class FineTunerReal:
    """Fine-Tuning real para modelos GGUF usando llama.cpp"""
    
    def __init__(self, ia):
        self.ia = ia
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.modelo_original = ia.config.modelo_path
        self.llama_dir = ia.config.llama_dir
        
        self.dataset_path = os.path.join(self.base_dir, "dataset_finetune.jsonl")
        self.modelo_finetuned = os.path.join(self.base_dir, "modelo_finetuned.gguf")
        
        self.esta_entrenando = False
        self.progreso = 0
        self.total_pares = 0
        
        self.finetune_exe = os.path.join(self.llama_dir, "finetune.exe")
        
        print(f"[FINETUNE] finetune.exe: {self.finetune_exe} (Existe: {os.path.exists(self.finetune_exe)})")
    
    def generar_dataset(self, contenido: str, nombre: str) -> Dict:
        """Genera dataset de entrenamiento"""
        resultado = {
            'success': False,
            'documento': nombre,
            'pares_generados': 0,
            'mensaje': ''
        }
        
        try:
            print(f"[FINETUNE] Generando dataset desde: {nombre}")
            
            pares = []
            
            # Extraer datos del JSON
            if 'mayoreo_todos_Volumen' in contenido:
                import re
                match = re.search(r'mayoreo_todos_Volumen:\s*([\d.]+)', contenido)
                if match:
                    volumen = match.group(1)
                    pares.append({
                        "instruction": "¿Cuál es el volumen total de mayoreo?",
                        "output": f"El volumen total de mayoreo es {volumen}."
                    })
                    pares.append({
                        "instruction": "Dame el total de mayoreo",
                        "output": f"El total de mayoreo es {volumen}."
                    })
                
                # Buscar periodos
                periodos = re.findall(r'"(agosto|julio|junio|mayo|abril|marzo)"', contenido, re.IGNORECASE)
                if periodos:
                    periodos_unicos = list(set([p.lower() for p in periodos]))
                    pares.append({
                        "instruction": "¿Qué periodos están disponibles?",
                        "output": f"Los periodos disponibles son: {', '.join(periodos_unicos)}."
                    })
            
            if not pares:
                pares.append({
                    "instruction": "¿Qué información contiene este documento?",
                    "output": contenido[:200]
                })
            
            # Guardar pares
            with open(self.dataset_path, 'a', encoding='utf-8') as f:
                for par in pares:
                    f.write(json.dumps(par, ensure_ascii=False) + '\n')
            
            self.total_pares += len(pares)
            resultado['pares_generados'] = len(pares)
            resultado['success'] = True
            resultado['mensaje'] = f"Generados {len(pares)} pares"
            
            print(f"[FINETUNE] Dataset generado: {len(pares)} pares")
            
            return resultado
            
        except Exception as e:
            resultado['mensaje'] = f"Error: {str(e)}"
            print(f"[FINETUNE ERROR] {e}")
            return resultado
    
    def iniciar_finetuning(self, callback=None) -> Dict:
        """Inicia el fine-tuning real"""
        if self.esta_entrenando:
            return {'success': False, 'mensaje': 'Ya hay un entrenamiento en progreso'}
        
        if not os.path.exists(self.finetune_exe):
            return {
                'success': False, 
                'mensaje': f'No se encuentra finetune.exe en: {self.finetune_exe}\nDescarga la version completa de llama.cpp'
            }
        
        if self.total_pares < 2:
            return {'success': False, 'mensaje': 'Se necesitan al menos 2 pares de entrenamiento'}
        
        self.esta_entrenando = True
        
        resultado = {
            'success': False,
            'total_pares': self.total_pares,
            'modelo_finetuned': self.modelo_finetuned,
            'mensaje': ''
        }
        
        def proceso():
            try:
                print("[FINETUNE] Iniciando fine-tuning...")
                
                comando = [
                    self.finetune_exe,
                    "--model-base", self.modelo_original,
                    "--train-data", self.dataset_path,
                    "--save", self.modelo_finetuned,
                    "--threads", "4",
                    "--epochs", "3",
                    "--learning-rate", "2e-5",
                    "--batch-size", "4",
                    "--ctx", "1024",
                ]
                
                print(f"[FINETUNE] Comando: {' '.join(comando)}")
                
                if callback:
                    callback(30, "Ejecutando fine-tuning...")
                
                proceso = subprocess.Popen(
                    comando,
                    cwd=self.llama_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                while True:
                    output = proceso.stdout.readline()
                    if output == '' and proceso.poll() is not None:
                        break
                    if output:
                        print(f"[FINETUNE] {output.strip()}")
                
                if proceso.returncode == 0 and os.path.exists(self.modelo_finetuned):
                    resultado['success'] = True
                    resultado['mensaje'] = f"Fine-tuning completado. Modelo guardado en: {self.modelo_finetuned}"
                    if callback:
                        callback(100, "¡Fine-tuning completado!")
                else:
                    stderr = proceso.stderr.read()
                    resultado['mensaje'] = f"Error: {stderr[:500]}"
                
                self.esta_entrenando = False
                
            except Exception as e:
                resultado['mensaje'] = f"Error: {str(e)}"
                self.esta_entrenando = False
        
        thread = threading.Thread(target=proceso)
        thread.daemon = True
        thread.start()
        
        return resultado
    
    def obtener_estado(self) -> Dict:
        return {
            'esta_entrenando': self.esta_entrenando,
            'progreso': self.progreso,
            'total_pares': self.total_pares,
            'modelo_finetuned': self.modelo_finetuned if os.path.exists(self.modelo_finetuned) else None
        }
    
    def limpiar_dataset(self):
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)
        self.total_pares = 0
        print("[FINETUNE] Dataset limpiado")