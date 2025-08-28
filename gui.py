import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from threading import Thread
import csv
from datetime import datetime
import time
import os

class TradingBotGUI:
    def __init__(self, root, bot_core):
        self.root = root
        self.bot = bot_core
        self.root.title("Simulador de bot comercial")
        self.root.geometry("1000x700")
        
        # Variables de control
        self.capital_inicial = tk.DoubleVar(value=50.0)
        self.capital_final = tk.DoubleVar(value=0.0)
        self.operaciones_realizadas = tk.IntVar(value=0)
        self.simulando = False
        self.historial = []
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Capital Inicial:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.capital_inicial).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(control_frame, text="Capital Final:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.capital_final).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(control_frame, text="Operaciones:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.operaciones_realizadas).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Button(control_frame, text="Iniciar Simulación", command=self.iniciar_simulacion).grid(row=3, column=0, pady=5)
        ttk.Button(control_frame, text="Detener", command=self.detener_simulacion).grid(row=3, column=1, pady=5)
        ttk.Button(control_frame, text="Ver Historial Completo", 
                  command=self.mostrar_historial_completo).grid(row=4, column=0, columnspan=2, pady=5)
        
        # Panel de gráficos
        graph_frame = ttk.LabelFrame(main_frame, text="Rendimiento", padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Panel de historial
        hist_frame = ttk.LabelFrame(main_frame, text="Historial Reciente", padding="10")
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tree = ttk.Treeview(hist_frame, columns=('Par', 'Retorno', 'Capital'), show='headings')
        self.tree.heading('Par', text='Par')
        self.tree.heading('Retorno', text='Retorno (%)')
        self.tree.heading('Capital', text='Capital')
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
    def mostrar_historial_completo(self):
        """Muestra el historial completo con manejo robusto de errores"""
        try:
            historial = self.bot.cargar_historial()
            
            if not historial:
                messagebox.showinfo("Historial", 
                    "No hay operaciones registradas o el historial estaba corrupto y se ha reiniciado.\n"
                    "Realice nuevas operaciones para generar datos.")
                return
            
            # Crear ventana de historial
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Historial Completo de Operaciones")
            hist_window.geometry("1100x650")
            
            # Frame principal
            main_frame = ttk.Frame(hist_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Treeview con scroll
            tree_frame = ttk.Frame(main_frame)
            tree_frame.pack(fill=tk.BOTH, expand=True)
            
            # Configurar columnas
            columns = ('timestamp', 'par', 'estrategia', 'retorno', 
                    'capital_antes', 'capital_despues', 'modelo_hash')
            
            tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
            
            # Configurar cabeceras
            tree.heading('timestamp', text='Fecha/Hora')
            tree.heading('par', text='Par')
            tree.heading('estrategia', text='Estrategia')
            tree.heading('retorno', text='Retorno (%)')
            tree.heading('capital_antes', text='Capital Antes')
            tree.heading('capital_despues', text='Capital Después')
            tree.heading('modelo_hash', text='Hash Modelo')
            
            # Configurar anchos de columna
            tree.column('timestamp', width=180, anchor='center')
            tree.column('par', width=100, anchor='center')
            tree.column('estrategia', width=150, anchor='center')
            tree.column('retorno', width=100, anchor='e')
            tree.column('capital_antes', width=120, anchor='e')
            tree.column('capital_despues', width=120, anchor='e')
            tree.column('modelo_hash', width=150, anchor='center')
            
            # Scrollbar
            scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscroll=scroll.set)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True)
            
            # Insertar datos con formato
            for op in reversed(historial):
                tree.insert('', 'end', values=(
                    op.get('timestamp', 'N/A'),
                    op.get('par', 'N/A'),
                    op.get('estrategia', 'N/A'),
                    f"{float(op.get('retorno', 0)):.2f}%",
                    f"${float(op.get('capital_antes', 0)):,.2f}",
                    f"${float(op.get('capital_despues', 0)):,.2f}",
                    op.get('modelo_hash', 'N/A')
                ))
            
            # Frame de botones
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            # Botón de exportar
            ttk.Button(btn_frame, text="Exportar a CSV", 
                    command=lambda: self.exportar_historial(historial)).pack(side=tk.LEFT, padx=5)
            
            # Botón de cerrar
            ttk.Button(btn_frame, text="Cerrar", 
                    command=hist_window.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", 
                f"No se pudo mostrar el historial:\n{str(e)}\n\n"
                "El archivo de historial ha sido regenerado.")
            # Regenerar archivo corrupto
            if hasattr(self.bot, 'historial_file') and os.path.exists(self.bot.historial_file):
                os.remove(self.bot.historial_file)
                self.bot.crear_estructura_historial()
    
    def exportar_historial(self, historial):
        """Exporta el historial completo a un nuevo archivo CSV"""
        if not historial:
            messagebox.showerror("Error", "No hay datos para exportar")
            return
        
        export_file = f"data/historial_exportado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with open(export_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=historial[0].keys())
                writer.writeheader()
                writer.writerows(historial)
            
            messagebox.showinfo("Éxito", f"Historial exportado a:\n{export_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar:\n{str(e)}")
        
    def actualizar_interfaz(self, operaciones, capital_final):
        # Actualizar capital final
        self.capital_final.set(round(capital_final, 2))
        
        # Agregar al historial
        self.historial.extend(operaciones)
        
        # Actualizar historial en tabla
        self.tree.delete(*self.tree.get_children())
        for op in self.historial[-20:]:  # Mostrar las últimas 20 operaciones
            self.tree.insert('', 'end', values=(
                op[0], 
                f"{op[1]:.2f}%", 
                f"${op[2]:.2f}"
            ))
        
        # Actualizar gráfico
        self.ax.clear()
        if len(self.historial) > 0:
            capital_history = [x[2] for x in self.historial]
            self.ax.plot(capital_history, label='Capital')
            self.ax.set_title('Evolución del Capital')
            self.ax.set_xlabel('Operación')
            self.ax.set_ylabel('Capital ($)')
            self.ax.legend()
            self.canvas.draw()
        
        # Actualizar contador
        self.operaciones_realizadas.set(len(self.historial))
        self.root.update()
    
    def iniciar_simulacion(self):
        if self.simulando:
            return
            
        self.simulando = True
        self.historial = []
        self.status_var.set("Iniciando simulación...")
        
        # Ejecutar en un hilo separado
        Thread(target=self.ejecutar_simulacion, daemon=True).start()
    
    def ejecutar_simulacion(self):
        try:
            # Iniciar simulación en el core
            hash_modelo = self.bot.iniciar_simulacion(self.capital_inicial.get())
            self.status_var.set(f"Modelo cargado. Hash: {hash_modelo}")
            
            # Realizar 10 operaciones
            for i in range(10):
                if not self.simulando:
                    break
                    
                operaciones, capital = self.bot.simular_operacion()
                self.actualizar_interfaz(operaciones, capital)
                self.status_var.set(f"Simulando... Operación {i+1}/10")
                
                # Pequeña pausa
                time.sleep(0.5)
            
            # Mostrar resumen
            self.status_var.set(f"Simulación completada. Capital final: ${capital:.2f}")
            messagebox.showinfo("Simulación completada", 
                               f"Capital inicial: ${self.capital_inicial.get():.2f}\n"
                               f"Capital final: ${capital:.2f}\n"
                               f"Operaciones realizadas: {self.operaciones_realizadas.get()}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error:\n{str(e)}")
            
        finally:
            self.simulando = False
    
    def detener_simulacion(self):
        self.simulando = False
        self.status_var.set("Simulación detenida por el usuario")