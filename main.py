from bot_core import TradingBotCore
from gui import TradingBotGUI
import tkinter as tk
import threading

def actualizar_datos_periodicamente_en_hilo(bot):
    # Ejecuta la actualización de datos periódicamente sin bloquear la interfaz
    bot.actualizar_datos_periodicamente(intervalo_segundos=5)

if __name__ == "__main__":
    # Crear instancia del core
    bot = TradingBotCore()
    #bot.generar_historial_fake()  # Agrega datos fake para entrenar
    bot.iniciar_simulacion(5000)
    
    # Crear un hilo para la actualización de datos
    hilo_actualizacion = threading.Thread(target=actualizar_datos_periodicamente_en_hilo, args=(bot,))
    hilo_actualizacion.daemon = True  # El hilo se cerrará cuando la aplicación termine
    hilo_actualizacion.start()
    
    # Crear interfaz gráfica
    root = tk.Tk()
    app = TradingBotGUI(root, bot)
    root.mainloop()
