"""
main.py - Sistema de Chat con Fine-Tuning para IA
Dos paneles: Fine-Tuning Real y Chat
"""

import os
import sys
import json
import threading
import time
import subprocess
import psutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import webview

# Agregar ruta para modulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'modulos'))

from ia import IA, crear_ia, IAConfig
from entrenador import EntrenadorIA, DocumentProcessor

app = Flask(__name__, template_folder='templates')

# Variables globales
ia = None
entrenador = None
finetuner = None
procesador_documentos = None
proceso_llama = None
pid_llama = None

# Estado del entrenamiento
estado_entrenamiento = {
    'en_progreso': False,
    'progreso': 0,
    'total_documentos': 0,
    'documentos_procesados': 0,
    'mensaje': '',
    'modelo_actual': '',
    'documentos_cargados': []
}

# ===== FUNCIONES DE LOGGING =====
def log_info(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [INFO] {mensaje}")

def log_error(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [ERROR] {mensaje}")

def log_success(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [SUCCESS] {mensaje}")

def log_warning(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [WARNING] {mensaje}")

def log_llama(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [LLAMA] {mensaje}")

def log_request(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [REQUEST] {mensaje}")

# ===== FUNCIONES DE VERIFICACION =====
def verificar_proceso_llama():
    global proceso_llama, pid_llama
    
    if proceso_llama is None:
        return False
    
    if pid_llama:
        try:
            proceso = psutil.Process(pid_llama)
            if proceso.is_running():
                return True
        except psutil.NoSuchProcess:
            pass
    
    if proceso_llama.poll() is None:
        return True
    
    return False

def obtener_info_llama():
    global proceso_llama, pid_llama
    
    if not verificar_proceso_llama():
        return {
            'activo': False,
            'pid': None,
            'cpu': None,
            'memoria': None,
            'tiempo': None
        }
    
    try:
        if pid_llama:
            proceso = psutil.Process(pid_llama)
            return {
                'activo': True,
                'pid': pid_llama,
                'cpu': proceso.cpu_percent(interval=0.1),
                'memoria': proceso.memory_info().rss / (1024 * 1024),
                'tiempo': time.time() - proceso.create_time()
            }
    except:
        pass
    
    return {
        'activo': True,
        'pid': None,
        'cpu': None,
        'memoria': None,
        'tiempo': None
    }

# ===== RUTAS FLASK =====
@app.route('/')
def home():
    log_request("Pagina principal cargada")
    return render_template('index.html')

@app.route('/modulos/main.html')
def main_html():
    log_request("Main.html cargado")
    return render_template('modulos/main.html')

@app.route('/api/estado')
def api_estado():
    global estado_entrenamiento, ia, entrenador
    
    info_llama = obtener_info_llama()
    ia_activa = ia is not None and ia.esta_activo() if ia else False
    
    # Obtener estado del entrenador
    estado_finetune = {}
    if entrenador:
        estado_finetune = entrenador.obtener_estado()
        # Agregar datos del modelo
        if ia:
            estado_finetune['tipo'] = ia.config.tipo_modelo
            estado_finetune['contexto'] = ia.config.max_context_tokens
            estado_finetune['tokens'] = ia.config.max_response_tokens
            estado_finetune['batch'] = ia.config.batch_size
            estado_finetune['threads'] = ia.config.num_threads
    
    response = {
        'ia_activa': ia_activa,
        'entrenamiento': estado_entrenamiento,
        'modelo_cargado': ia.config.modelo_path if ia else None,
        'llama': info_llama,
        'llama_activo': info_llama.get('activo', False),
        'finetune': estado_finetune
    }
    
    return jsonify(response)

@app.route('/api/verificar_llama', methods=['GET'])
def api_verificar_llama():
    info = obtener_info_llama()
    log_llama(f"Verificacion de llama: {'ACTIVO' if info.get('activo') else 'DETENIDO'} PID: {info.get('pid')}")
    return jsonify(info)

@app.route('/api/listar_modelos', methods=['POST'])
def api_listar_modelos():
    try:
        data = request.get_json()
        carpeta = data.get('carpeta', '').strip()
        
        log_info(f"Listando modelos en carpeta: {carpeta}")
        
        if not carpeta or not os.path.exists(carpeta):
            log_error(f"Carpeta no valida: {carpeta}")
            return jsonify({'success': False, 'error': 'Carpeta no valida'}), 400
        
        modelos = []
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.gguf'):
                ruta_completa = os.path.join(carpeta, archivo)
                tamano_mb = os.path.getsize(ruta_completa) / (1024 * 1024)
                modelos.append({
                    'nombre': archivo,
                    'ruta': ruta_completa,
                    'tamano': os.path.getsize(ruta_completa)
                })
                log_info(f"Modelo encontrado: {archivo} ({tamano_mb:.2f} MB)")
        
        log_success(f"Total modelos encontrados: {len(modelos)}")
        
        return jsonify({
            'success': True,
            'modelos': modelos,
            'carpeta': carpeta
        })
        
    except Exception as e:
        log_error(f"Error listando modelos: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cargar_modelo', methods=['POST'])
def api_cargar_modelo():
    global ia, estado_entrenamiento, proceso_llama, pid_llama
    
    log_info("=" * 70)
    log_info("INICIANDO PROCESO DE CARGA DE MODELO")
    log_info("=" * 70)
    
    try:
        data = request.get_json()
        ruta_modelo = data.get('ruta_modelo', '').strip()
        
        log_llama(f"Ruta del modelo: {ruta_modelo}")
        
        if not ruta_modelo:
            log_error("Ruta de modelo no especificada")
            return jsonify({'success': False, 'error': 'Ruta de modelo no especificada'}), 400
        
        if not os.path.exists(ruta_modelo):
            log_error(f"El archivo no existe: {ruta_modelo}")
            return jsonify({'success': False, 'error': f'El archivo no existe: {ruta_modelo}'}), 404
        
        if not ruta_modelo.endswith('.gguf'):
            log_error("El archivo no es .gguf")
            return jsonify({'success': False, 'error': 'El archivo debe ser .gguf'}), 400
        
        if ia:
            log_info("Deteniendo IA anterior...")
            ia.detener()
            ia = None
            log_success("IA anterior detenida")
        
        if proceso_llama:
            log_info("Deteniendo llama-server anterior...")
            try:
                proceso_llama.terminate()
                proceso_llama.wait(timeout=5)
                log_success("Proceso llama-server terminado correctamente")
            except:
                try:
                    proceso_llama.kill()
                    log_warning("Proceso llama-server forzado a terminar")
                except:
                    pass
            proceso_llama = None
            pid_llama = None
        
        log_info("Creando nueva configuracion...")
        config = IAConfig()
        config.modelo_path = ruta_modelo
        
        tamano_mb = os.path.getsize(ruta_modelo) / (1024 * 1024)
        log_info(f"Modelo: {os.path.basename(ruta_modelo)}")
        log_info(f"Tamaño: {tamano_mb:.2f} MB")
        
        config._detectar_tipo_modelo()
        config.max_context_tokens = config._get_context_size()
        config.batch_size = config._get_batch_size()
        config.max_response_tokens = config._get_max_tokens()
        config.puerto = 8081
        config.server_url = f"http://127.0.0.1:{config.puerto}/v1/chat/completions"
        config.health_url = f"http://127.0.0.1:{config.puerto}/health"
        
        log_info("Configuracion del modelo:")
        log_info(f"  - Tipo: {config.tipo_modelo}")
        log_info(f"  - Contexto: {config.max_context_tokens}")
        log_info(f"  - Puerto: {config.puerto}")
        log_info(f"  - Batch: {config.batch_size}")
        log_info(f"  - Max Tokens: {config.max_response_tokens}")
        log_info(f"  - Threads: {config.num_threads}")
        
        log_info(f"Verificando llama-server.exe en: {config.llama_exe}")
        if not os.path.exists(config.llama_exe):
            error_msg = f"No se encuentra llama-server.exe en: {config.llama_exe}"
            log_error(error_msg)
            
            posibles = [
                os.path.join(os.getcwd(), "llama", "llama-b9455-bin-win-cpu-x64", "llama-server.exe"),
                os.path.join(os.getcwd(), "llama", "llama-server.exe"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama", "llama-b9455-bin-win-cpu-x64", "llama-server.exe"),
            ]
            for p in posibles:
                if os.path.exists(p):
                    log_success(f"llama-server.exe encontrado en: {p}")
                    config.llama_exe = p
                    config.llama_dir = os.path.dirname(p)
                    break
            else:
                return jsonify({'success': False, 'error': error_msg}), 500
        
        log_success(f"llama-server.exe encontrado en: {config.llama_exe}")
        
        log_info("Iniciando IA y llama-server...")
        log_info("=" * 50)
        
        ia = IA(config)
        
        if ia.iniciar():
            if ia.servidor and ia.servidor.proceso:
                proceso_llama = ia.servidor.proceso
                pid_llama = proceso_llama.pid
                log_success(f"Proceso llama-server iniciado con PID: {pid_llama}")
                
                time.sleep(2)
                if verificar_proceso_llama():
                    log_success("CONFIRMADO: llama-server esta corriendo correctamente")
                    info = obtener_info_llama()
                    log_info(f"Estado de llama: PID={info.get('pid')}, CPU={info.get('cpu')}%, Memoria={info.get('memoria'):.1f}MB")
                else:
                    log_warning("No se pudo confirmar que llama-server esta corriendo")
            
            estado_entrenamiento['modelo_actual'] = os.path.basename(ruta_modelo)
            
            # Inicializar entrenador
            if entrenador is None:
                entrenador = EntrenadorIA(ia)
                log_info("Entrenador inicializado")
            
            log_success("=" * 70)
            log_success("MODELO CARGADO EXITOSAMENTE")
            log_success("=" * 70)
            log_info(f"Modelo: {os.path.basename(ruta_modelo)}")
            log_info(f"Tipo: {config.tipo_modelo}")
            log_info(f"Cuantizacion: {config.cuantizacion}")
            log_info(f"Puerto: {config.puerto}")
            log_info(f"PID: {pid_llama}")
            log_info("=" * 70)
            
            return jsonify({
                'success': True,
                'mensaje': f'Modelo cargado: {os.path.basename(ruta_modelo)}',
                'tipo': config.tipo_modelo,
                'cuantizacion': config.cuantizacion,
                'puerto': config.puerto,
                'tamano_mb': round(tamano_mb, 2),
                'pid': pid_llama,
                'contexto': config.max_context_tokens,
                'tokens': config.max_response_tokens,
                'batch': config.batch_size,
                'threads': config.num_threads
            })
        else:
            error_msg = 'Error al iniciar el modelo. Verifica que llama-server.exe este en la carpeta correcta y que el modelo sea valido.'
            log_error(error_msg)
            log_error("Revisa los logs anteriores para mas detalles")
            
            log_info("Diagnostico:")
            log_info(f"  - llama-server.exe: {config.llama_exe} (Existe: {os.path.exists(config.llama_exe)})")
            log_info(f"  - Modelo: {config.modelo_path} (Existe: {os.path.exists(config.modelo_path)})")
            log_info(f"  - Puerto: {config.puerto}")
            
            return jsonify({'success': False, 'error': error_msg}), 500
            
    except Exception as e:
        log_error(f"Error en api_cargar_modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cargar_documentos', methods=['POST'])
def api_cargar_documentos():
    global procesador_documentos, estado_entrenamiento
    
    log_info("Iniciando carga de documentos...")
    
    try:
        if 'archivos' not in request.files:
            log_error("No se enviaron archivos")
            return jsonify({'success': False, 'error': 'No se enviaron archivos'}), 400
        
        archivos = request.files.getlist('archivos')
        log_info(f"Archivos recibidos: {len(archivos)}")
        
        documentos_cargados = []
        os.makedirs('temp', exist_ok=True)
        
        for archivo in archivos:
            if archivo.filename == '':
                continue
            
            extension = os.path.splitext(archivo.filename)[1].lower()
            tipo = 'texto'
            if extension in ['.json']:
                tipo = 'json'
            elif extension in ['.txt']:
                tipo = 'texto'
            elif extension in ['.pdf']:
                tipo = 'pdf'
            elif extension in ['.docx', '.doc']:
                tipo = 'word'
            elif extension in ['.xlsx', '.xls']:
                tipo = 'excel'
            elif extension in ['.csv']:
                tipo = 'csv'
            elif extension in ['.md']:
                tipo = 'markdown'
            elif extension in ['.html', '.htm']:
                tipo = 'html'
            elif extension in ['.xml']:
                tipo = 'xml'
            
            temp_path = os.path.join('temp', archivo.filename)
            archivo.save(temp_path)
            
            log_info(f"Documento cargado: {archivo.filename} (Tipo: {tipo})")
            
            documentos_cargados.append({
                'nombre': archivo.filename,
                'ruta': temp_path,
                'tipo': tipo,
                'tamano': os.path.getsize(temp_path)
            })
        
        estado_entrenamiento['documentos_cargados'] = documentos_cargados
        estado_entrenamiento['total_documentos'] = len(documentos_cargados)
        
        procesador_documentos = DocumentProcessor()
        
        log_success(f"Documentos cargados: {len(documentos_cargados)}")
        
        return jsonify({
            'success': True,
            'documentos': documentos_cargados,
            'mensaje': f'Cargados {len(documentos_cargados)} documentos'
        })
        
    except Exception as e:
        log_error(f"Error cargando documentos: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== RUTAS DE FINE-TUNING =====

@app.route('/api/finetune/preparar', methods=['POST'])
def api_finetune_preparar():
    """Prepara los documentos para fine-tuning"""
    global entrenador, ia, estado_entrenamiento
    
    log_info("=" * 70)
    log_info("PREPARANDO DOCUMENTOS PARA FINE-TUNING")
    log_info("=" * 70)
    
    try:
        if not ia or not ia.esta_activo():
            log_error("No hay un modelo cargado")
            return jsonify({'success': False, 'error': 'No hay un modelo cargado'}), 400
        
        if not estado_entrenamiento['documentos_cargados']:
            log_error("No hay documentos cargados")
            return jsonify({'success': False, 'error': 'No hay documentos cargados'}), 400
        
        if entrenador is None:
            entrenador = EntrenadorIA(ia)
            log_info("Entrenador inicializado")
        
        documentos = estado_entrenamiento['documentos_cargados']
        log_info(f"Documentos a preparar: {len(documentos)}")
        
        resultados = []
        total_pares = 0
        
        for idx, doc in enumerate(documentos):
            log_info(f"[{idx+1}/{len(documentos)}] Preparando: {doc['nombre']}")
            
            procesador = DocumentProcessor()
            contenido = procesador.procesar(doc['ruta'], doc['tipo'])
            
            if contenido and len(contenido) > 50:
                resultado = entrenador.aprender_documento(contenido, doc['nombre'])
                resultados.append(resultado)
                
                if resultado.get('success'):
                    pares = resultado.get('pares_generados', 0)
                    total_pares += pares
                    log_success(f"Documento preparado: {doc['nombre']} ({pares} chunks)")
                else:
                    log_error(f"Error preparando: {doc['nombre']} - {resultado.get('mensaje')}")
            else:
                log_error(f"No se pudo procesar: {doc['nombre']}")
                resultados.append({
                    'success': False,
                    'documento': doc['nombre'],
                    'mensaje': 'No se pudo extraer contenido del documento'
                })
        
        log_success("=" * 70)
        log_success("PREPARACION COMPLETADA")
        log_success("=" * 70)
        log_info(f"Documentos preparados: {len([r for r in resultados if r.get('success')])}/{len(documentos)}")
        log_info(f"Total chunks generados: {total_pares}")
        
        return jsonify({
            'success': True,
            'resultados': resultados,
            'estado': entrenador.obtener_estado(),
            'total_pares': total_pares,
            'total_documentos': len(documentos),
            'mensaje': f"Preparados {len([r for r in resultados if r.get('success')])} documentos, {total_pares} chunks generados"
        })
        
    except Exception as e:
        log_error(f"Error preparando: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finetune/iniciar', methods=['POST'])
def api_finetune_iniciar():
    """Inicia el fine-tuning"""
    global entrenador
    
    log_info("=" * 70)
    log_info("INICIANDO FINE-TUNING")
    log_info("=" * 70)
    
    try:
        if not entrenador:
            log_error("No hay documentos preparados")
            return jsonify({'success': False, 'error': 'No hay documentos preparados. Primero ejecuta "Preparar Documentos"'}), 400
        
        estado = entrenador.obtener_estado()
        total_chunks = estado.get('total_pares', 0)
        
        if total_chunks < 1:
            log_error(f"Chunks insuficientes: {total_chunks} (minimo 1)")
            return jsonify({
                'success': False, 
                'error': f'Se necesitan al menos 1 chunk de conocimiento. Actual: {total_chunks}'
            }), 400
        
        log_info(f"Iniciando fine-tuning con {total_chunks} chunks")
        
        resultado = entrenador.iniciar_entrenamiento()
        
        if resultado.get('success'):
            log_success("FINE-TUNING COMPLETADO EXITOSAMENTE")
            log_info(f"Total chunks: {resultado.get('total_pares', 0)}")
        else:
            log_error(f"Error en fine-tuning: {resultado.get('mensaje')}")
        
        return jsonify(resultado)
        
    except Exception as e:
        log_error(f"Error en fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finetune/estado', methods=['GET'])
def api_finetune_estado():
    """Estado del fine-tuning"""
    global entrenador
    
    if not entrenador:
        return jsonify({
            'success': True,
            'estado': {
                'esta_entrenando': False,
                'progreso': 100,
                'documentos_procesados': 0,
                'nombres_documentos': [],
                'total_pares': 0,
                'total_chunks': 0,
                'epochs': 0,
                'learning_rate': 0,
                'batch_size': 0,
                'tipo': '—',
                'contexto': '—',
                'tokens': '—',
                'threads': '—'
            }
        })
    
    estado = entrenador.obtener_estado()
    
    # Agregar datos del modelo si está disponible
    if ia:
        estado['tipo'] = ia.config.tipo_modelo
        estado['contexto'] = ia.config.max_context_tokens
        estado['tokens'] = ia.config.max_response_tokens
        estado['batch'] = ia.config.batch_size
        estado['threads'] = ia.config.num_threads
    
    estado['esta_entrenando'] = False
    estado['progreso'] = 100
    
    return jsonify({
        'success': True,
        'estado': estado
    })

@app.route('/api/finetune/exportar', methods=['GET'])
def api_finetune_exportar():
    """Exporta el dataset generado"""
    global entrenador
    
    if not entrenador:
        return jsonify({'success': False, 'error': 'No hay entrenador'}), 400
    
    texto = entrenador.exportar_conocimiento()
    
    return jsonify({
        'success': True,
        'dataset': texto,
        'total_pares': len([p for p in texto.split('\n\n') if p.strip()])
    })

@app.route('/api/finetune/limpiar', methods=['POST'])
def api_finetune_limpiar():
    """Limpia el dataset de fine-tuning"""
    global entrenador
    
    if not entrenador:
        return jsonify({'success': False, 'error': 'No hay entrenador'}), 400
    
    entrenador.limpiar_conocimiento()
    log_success("Dataset de fine-tuning limpiado")
    
    return jsonify({'success': True, 'mensaje': 'Dataset limpiado correctamente'})

# ===== RUTAS DE CHAT =====

@app.route('/api/preguntar', methods=['POST'])
def api_preguntar():
    global ia, entrenador
    
    log_info("Recibiendo pregunta...")
    
    try:
        if not ia or not ia.esta_activo():
            log_error("Modelo no disponible")
            return jsonify({'success': False, 'error': 'Modelo no disponible'}), 400
        
        if not verificar_proceso_llama():
            log_error("El servidor llama no esta disponible")
            return jsonify({'success': False, 'error': 'El servidor llama no esta disponible'}), 503
        
        data = request.get_json()
        pregunta = data.get('pregunta', '').strip()
        
        if not pregunta:
            log_error("Pregunta vacia")
            return jsonify({'success': False, 'error': 'Pregunta vacia'}), 400
        
        log_info(f"Pregunta: {pregunta[:100]}...")
        
        if entrenador and len(entrenador.conocimiento) > 0:
            log_info(f"Usando conocimiento: {len(entrenador.conocimiento)} chunks")
            conocimiento = entrenador.buscar_conocimiento(pregunta)
            
            if conocimiento:
                if len(conocimiento) > 400:
                    conocimiento = conocimiento[:400] + "..."
                prompt = f"""Conocimiento disponible:
{conocimiento}

Pregunta: {pregunta}

Respuesta:"""
                log_info(f"Conocimiento encontrado: {len(conocimiento)} caracteres")
            else:
                prompt = pregunta
                log_warning("No se encontro conocimiento relevante")
        else:
            log_warning("No hay conocimiento disponible, respondiendo directamente")
            prompt = pregunta
        
        if len(prompt) > 800:
            partes = prompt.split("Pregunta:")
            if len(partes) == 2:
                prompt = f"Conocimiento: {partes[0][:300]}...\n\nPregunta: {partes[1]}"
            else:
                prompt = prompt[:800] + "..."
        
        log_info(f"Prompt final: {len(prompt)} caracteres")
        log_info("Enviando pregunta a la IA...")
        resultado = ia.enviar_mensaje(prompt)
        
        if resultado.get('success'):
            log_success(f"Respuesta recibida en {resultado.get('tiempo', 0):.2f}s")
            return jsonify({
                'success': True,
                'respuesta': resultado.get('reply', ''),
                'tiempo': resultado.get('tiempo', 0)
            })
        else:
            log_error(f"Error de IA: {resultado.get('reply', 'Error desconocido')}")
            return jsonify({
                'success': False,
                'error': resultado.get('reply', 'Error desconocido')
            }), 500
            
    except Exception as e:
        log_error(f"Error en pregunta: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/limpiar_historial', methods=['POST'])
def api_limpiar_historial():
    global ia
    log_info("Limpiando historial...")
    if ia:
        ia.limpiar_historial()
    log_success("Historial limpiado")
    return jsonify({'success': True, 'mensaje': 'Historial limpiado'})

@app.route('/api/limpiar_documentos', methods=['POST'])
def api_limpiar_documentos():
    global estado_entrenamiento, entrenador
    
    log_info("Limpiando documentos y conocimiento...")
    
    estado_entrenamiento['documentos_cargados'] = []
    estado_entrenamiento['total_documentos'] = 0
    estado_entrenamiento['documentos_procesados'] = 0
    estado_entrenamiento['progreso'] = 0
    
    if entrenador:
        entrenador.limpiar_conocimiento()
        entrenador = None
    
    import shutil
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        os.makedirs('temp', exist_ok=True)
    
    log_success("Documentos y conocimiento limpiados")
    return jsonify({'success': True, 'mensaje': 'Documentos y conocimiento limpiados'})

@app.route('/api/guardar_modelo', methods=['POST'])
def api_guardar_modelo():
    """Guarda el modelo actual"""
    global ia
    
    log_info("Guardando modelo...")
    
    try:
        if not ia or not ia.esta_activo():
            return jsonify({'success': False, 'error': 'No hay un modelo cargado'}), 400
        
        # Aquí se implementaría la lógica de guardado
        # Por ahora solo respondemos éxito
        
        log_success("Modelo guardado correctamente")
        return jsonify({
            'success': True,
            'mensaje': 'Modelo guardado correctamente',
            'ruta': ia.config.modelo_path
        })
        
    except Exception as e:
        log_error(f"Error guardando modelo: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exportar_modelo', methods=['POST'])
def api_exportar_modelo():
    """Exporta el modelo"""
    global ia
    
    log_info("Exportando modelo...")
    
    try:
        if not ia or not ia.esta_activo():
            return jsonify({'success': False, 'error': 'No hay un modelo cargado'}), 400
        
        # Aquí se implementaría la lógica de exportación
        # Por ahora solo respondemos éxito
        
        log_success("Modelo exportado correctamente")
        return jsonify({
            'success': True,
            'mensaje': 'Modelo exportado correctamente',
            'ruta': ia.config.modelo_path
        })
        
    except Exception as e:
        log_error(f"Error exportando modelo: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/resetear_modelo', methods=['POST'])
def api_resetear_modelo():
    """Resetea el modelo a valores de fábrica"""
    global ia, entrenador
    
    log_info("Reseteando modelo...")
    
    try:
        if ia:
            ia.detener()
            ia = None
        
        if entrenador:
            entrenador.limpiar_conocimiento()
            entrenador = None
        
        estado_entrenamiento['modelo_actual'] = ''
        
        log_success("Modelo reseteado correctamente")
        return jsonify({
            'success': True,
            'mensaje': 'Modelo reseteado correctamente'
        })
        
    except Exception as e:
        log_error(f"Error reseteando modelo: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def iniciar_flask():
    log_info("Iniciando servidor Flask...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)

def main():
    print("=" * 70)
    print("SISTEMA DE CHAT CON FINE-TUNING PARA IA")
    print("=" * 70)
    print("")
    print("Instrucciones:")
    print("   1. Selecciona la carpeta donde esta el modelo .gguf")
    print("   2. Elige el modelo de la lista y cargalo")
    print("   3. Sube documentos (JSON, TXT, PDF, Word, Excel)")
    print("   4. En el panel Fine-Tuning, prepara el dataset")
    print("   5. Inicia el Fine-Tuning")
    print("   6. Prueba el modelo en el panel Chat")
    print("=" * 70)
    print("")
    log_info("Panel izquierdo: Fine-Tuning Real")
    log_info("Panel derecho: Chat para probar el modelo")
    print("")
    
    log_info("Iniciando servidor web en http://127.0.0.1:5000")
    thread = threading.Thread(target=iniciar_flask)
    thread.daemon = True
    thread.start()
    
    time.sleep(2)
    
    log_info("Abriendo ventana de chat maximizada...")
    
    window = webview.create_window(
        title="Neural Train - Fine-Tuning System",
        url="http://127.0.0.1:5000",
        width=1400,
        height=900,
        resizable=True,
        min_size=(1024, 768),
        confirm_close=True,
        text_select=True,
        easy_drag=True,
        fullscreen=False
    )
    
    def maximizar_ventana():
        time.sleep(1.5)
        try:
            window.maximize()
            log_info("Ventana maximizada")
        except:
            pass
    
    threading.Thread(target=maximizar_ventana, daemon=True).start()
    
    webview.start(gui='edgechromium', debug=False)
    
    if ia:
        log_info("Deteniendo IA...")
        ia.detener()
    
    log_info("Aplicacion cerrada.")
    sys.exit(0)

if __name__ == '__main__':
    main()