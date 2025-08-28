import ccxt
import pandas as pd
import numpy as np
import hashlib
import os
import time
import pickle
import csv
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBotEnhanced:
    def __init__(self, config_file: str = 'config.json'):
        """Inicializa el bot de trading con configuración desde archivo"""
        # Cargar configuración
        self.config = self._load_config(config_file)
        
        # Configuración de exchange
        self.exchange = self._setup_exchange()
        self.exchange.load_markets()
        
        # Parámetros de trading
        self.capital = self.config.get('initial_capital', 1000.0)
        self.monedas = self.config.get('trading_pairs', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'])
        self.timeframe = self.config.get('timeframe', '5m')
        
        # Estrategias
        self.estrategias = self._setup_strategies()
        
        # Modelo de ML
        self.modelo = None
        self.modelo_hash = ""
        self.features = []
        
        # Configuración de archivos
        self.data_dir = 'data'
        self.historial_file = os.path.join(self.data_dir, 'historial_operaciones.csv')
        self.model_dir = 'models'
        self._ensure_directories_exist()
        
        # Cargar modelo si existe
        self._load_model()
        
        logger.info("TradingBot inicializado correctamente")

    def _load_config(self, config_file: str) -> Dict:
        """Carga la configuración desde un archivo JSON"""
        try:
            with open(config_file) as f:
                config = json.load(f)
            logger.info("Configuración cargada correctamente")
            return config
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise

    def _setup_exchange(self):
        """Configura la conexión con el exchange"""
        try:
            exchange_class = getattr(ccxt, self.config.get('exchange', 'binance'))
            return exchange_class({
                'apiKey': self.config.get('api_key', ''),
                'secret': self.config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True
                }
            })
        except Exception as e:
            logger.error(f"Error configurando exchange: {e}")
            raise

    def _setup_strategies(self) -> Dict:
        """Configura las estrategias de trading disponibles"""
        strategies = {
            'Media Retornos': self._estrategia_media_retornos,
            'Media Movil 5': self._estrategia_media_movil_5,
            'RSI': self._estrategia_rsi,
            'MACD': self._estrategia_macd,
            'Bollinger': self._estrategia_bollinger
        }
        logger.info(f"Estrategias cargadas: {list(strategies.keys())}")
        return strategies

    def _ensure_directories_exist(self):
        """Asegura que los directorios necesarios existan"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self._crear_estructura_historial()

    def _crear_estructura_historial(self):
        """Crea el archivo de historial con cabeceras si no existe"""
        if not os.path.exists(self.historial_file):
            with open(self.historial_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'par', 'estrategia', 'retorno', 
                    'capital_antes', 'capital_despues', 'modelo_hash',
                    'metadata'
                ])
            logger.info("Archivo de historial creado")

    # ------------------------------------------
    # ESTRATEGIAS DE TRADING
    # ------------------------------------------
    
    def _estrategia_media_retornos(self, df: pd.DataFrame) -> float:
        """Estrategia básica: media de retornos históricos"""
        df = df.copy()
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        return df['returns'].mean()

    def _estrategia_media_movil_5(self, df: pd.DataFrame) -> float:
        """Estrategia de media móvil de 5 periodos"""
        df = df.copy()
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        return df['returns'].rolling(5).mean().iloc[-1]

    def _estrategia_rsi(self, df: pd.DataFrame, periodos: int = 14) -> float:
        """Estrategia RSI (Relative Strength Index)"""
        delta = df['close'].diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        
        avg_ganancia = ganancia.rolling(periodos).mean()
        avg_perdida = perdida.rolling(periodos).mean()
        
        rs = avg_ganancia / avg_perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _estrategia_macd(self, df: pd.DataFrame) -> float:
        """Estrategia MACD (Moving Average Convergence Divergence)"""
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd.iloc[-1] - signal.iloc[-1])

    def _estrategia_bollinger(self, df: pd.DataFrame, window: int = 20) -> float:
        """Estrategia Bandas de Bollinger"""
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # Normalizar la distancia al precio actual
        current_price = df['close'].iloc[-1]
        distance_to_upper = (upper_band.iloc[-1] - current_price) / current_price
        distance_to_lower = (current_price - lower_band.iloc[-1]) / current_price
        
        return distance_to_upper - distance_to_lower  # Positivo si cerca de banda inferior

    # ------------------------------------------
    # MANEJO DE DATOS
    # ------------------------------------------
    
    def obtener_datos(self, par: str, timeframe: str = None, limit: int = 1000) -> pd.DataFrame:
        """Obtiene datos OHLCV del exchange o del cache local"""
        timeframe = timeframe or self.timeframe
        nombre_archivo = self._nombre_archivo_datos(par, timeframe)
        
        # Verificar si tenemos datos recientes en cache
        if os.path.exists(nombre_archivo):
            file_age = time.time() - os.path.getmtime(nombre_archivo)
            if file_age < self.config.get('data_refresh_interval', 900):  # 15 minutos por defecto
                logger.info(f"Usando datos en cache para {par} ({timeframe})")
                return pd.read_csv(nombre_archivo)
        
        # Descargar nuevos datos si no existen o están desactualizados
        logger.info(f"Descargando datos para {par} ({timeframe})...")
        try:
            since = self.exchange.milliseconds() - 86400000 * 3  # 3 días de datos
            ohlcv = self.exchange.fetch_ohlcv(par, timeframe, since=since, limit=limit)
            
            if len(ohlcv) == 0:
                logger.warning(f"No se recibieron datos para {par}")
                if os.path.exists(nombre_archivo):  # Usar datos antiguos si existen
                    return pd.read_csv(nombre_archivo)
                raise ValueError(f"No hay datos disponibles para {par}")
            
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Calcular features adicionales
            df = self._calcular_features(df)
            
            # Guardar datos
            df.to_csv(nombre_archivo, index=False)
            logger.info(f"Datos guardados en {nombre_archivo}")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos para {par}: {e}")
            if os.path.exists(nombre_archivo):  # Fallback a datos locales
                logger.warning("Usando datos locales como fallback")
                return pd.read_csv(nombre_archivo)
            raise

    def _nombre_archivo_datos(self, par: str, timeframe: str) -> str:
        """Genera nombre de archivo seguro para guardar datos"""
        safe_pair = par.replace('/', '-').replace('\\', '-')
        return os.path.join(self.data_dir, f"{safe_pair}_{timeframe}.csv")

    def _calcular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos para el DataFrame"""
        # Retornos
        df['returns'] = df['close'].pct_change()
        
        # Medias móviles
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_upper'] = df['ma20'] + (2 * df['close'].rolling(20).std())
        df['bb_lower'] = df['ma20'] - (2 * df['close'].rolling(20).std())
        
        return df.dropna()

    # ------------------------------------------
    # BACKTESTING Y SIMULACIÓN
    # ------------------------------------------
    
    def backtest_estrategia(self, par: str, estrategia: str, capital_inicial: float = 1000) -> Dict:
        """Realiza backtesting de una estrategia en un par específico"""
        logger.info(f"Iniciando backtest para {par} con estrategia {estrategia}")
        
        df = self.obtener_datos(par)
        if len(df) < 100:
            raise ValueError(f"No hay suficientes datos para {par} (solo {len(df)} registros)")
        
        resultados = []
        capital = capital_inicial
        operaciones = []
        drawdowns = []
        max_capital = capital_inicial
        
        for i in range(30, len(df)):  # Empezar después de tener suficientes datos
            df_window = df.iloc[:i]
            signal = self.estrategias[estrategia](df_window)
            
            # Lógica de trading simple
            if signal > 0:  # Señal de compra
                retorno = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
                capital *= (1 + retorno)
                operaciones.append(retorno)
            
            # Calcular métricas
            resultados.append(capital)
            max_capital = max(max_capital, capital)
            drawdown = (capital - max_capital) / max_capital
            drawdowns.append(drawdown)
        
        # Calcular métricas finales
        retorno_total = (capital / capital_inicial - 1) * 100
        max_drawdown = min(drawdowns) * 100 if drawdowns else 0
        sharpe = self._calcular_sharpe(resultados)
        win_rate = len([x for x in operaciones if x > 0]) / len(operaciones) * 100 if operaciones else 0
        
        resultado = {
            'estrategia': estrategia,
            'par': par,
            'retorno_total': retorno_total,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_operaciones': len(operaciones),
            'capital_final': capital
        }
        
        logger.info(f"Backtest completado: {resultado}")
        return resultado

    def _calcular_sharpe(self, resultados: List[float], risk_free_rate: float = 0.0) -> float:
        """Calcula el ratio de Sharpe para una serie de resultados"""
        if len(resultados) < 2:
            return 0.0
        
        retornos = np.diff(resultados) / resultados[:-1]
        excess_returns = retornos - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365*24*12)  # Ajustado para timeframe 5m

    def _calcular_drawdown(self, resultados: List[float]) -> float:
        """Calcula el máximo drawdown para una serie de resultados"""
        peak = resultados[0]
        max_drawdown = 0.0
        
        for result in resultados:
            if result > peak:
                peak = result
            drawdown = (peak - result) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown * 100  # En porcentaje

    def comparar_estrategias(self, capital_inicial: float = 1000) -> pd.DataFrame:
        """Compara el rendimiento de todas las estrategias en todos los pares"""
        resultados = []
        
        for par in self.monedas:
            for estrategia in self.estrategias:
                try:
                    resultado = self.backtest_estrategia(par, estrategia, capital_inicial)
                    resultados.append(resultado)
                except Exception as e:
                    logger.error(f"Error en backtest para {par} con {estrategia}: {e}")
        
        df_resultados = pd.DataFrame(resultados)
        if not df_resultados.empty:
            df_resultados = df_resultados.sort_values('sharpe_ratio', ascending=False)
            logger.info("\nResultados comparativos:\n" + str(df_resultados))
        
        return df_resultados

    # ------------------------------------------
    # MODELO DE MACHINE LEARNING
    # ------------------------------------------
    
    def entrenar_modelo(self, X: np.array, y: np.array) -> RandomForestClassifier:
        """Entrena un modelo de clasificación con los datos proporcionados"""
        logger.info("Entrenando modelo...")
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Usar Random Forest en lugar de Regresión Logística
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        
        modelo.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = modelo.predict(X_test)
        logger.info("\nReporte de clasificación:\n" + classification_report(y_test, y_pred))
        
        # Guardar features importantes
        self.features = list(zip(
            list(self.estrategias.keys()),
            modelo.feature_importances_
        ))
        logger.info(f"Features importantes: {self.features}")
        
        return modelo

    def generar_hash_modelo(self, modelo) -> str:
        """Genera un hash único para el modelo actual"""
        if hasattr(modelo, 'feature_importances_'):
            resumen = str(modelo.feature_importances_).encode('utf-8')
        else:
            resumen = str(modelo.coef_).encode('utf-8')
        return hashlib.md5(resumen).hexdigest()

    def guardar_modelo(self, modelo) -> None:
        """Guarda el modelo entrenado en disco"""
        modelo_path = os.path.join(self.model_dir, 'modelo.pkl')
        hash_path = os.path.join(self.model_dir, 'modelo_hash.txt')
        
        try:
            with open(modelo_path, 'wb') as f:
                pickle.dump(modelo, f)
            
            self.modelo_hash = self.generar_hash_modelo(modelo)
            with open(hash_path, 'w') as f:
                f.write(self.modelo_hash)
            
            logger.info(f"Modelo guardado en {modelo_path} (hash: {self.modelo_hash})")
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise

    def _load_model(self) -> bool:
        """Carga el modelo desde disco si existe"""
        modelo_path = os.path.join(self.model_dir, 'modelo.pkl')
        hash_path = os.path.join(self.model_dir, 'modelo_hash.txt')
        
        if os.path.exists(modelo_path) and os.path.exists(hash_path):
            try:
                with open(modelo_path, 'rb') as f:
                    modelo = pickle.load(f)
                
                with open(hash_path, 'r') as f:
                    hash_guardado = f.read().strip()
                
                hash_calculado = self.generar_hash_modelo(modelo)
                
                if hash_guardado == hash_calculado:
                    self.modelo = modelo
                    self.modelo_hash = hash_guardado
                    logger.info("Modelo cargado correctamente desde disco")
                    return True
                else:
                    logger.warning("Hash del modelo no coincide - no se cargará")
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
        
        return False

    def entrenar_con_historial(self) -> None:
        """Entrena el modelo usando el historial de operaciones"""
        historial = self.cargar_historial()
        if not historial:
            logger.warning("No hay historial suficiente para entrenar")
            return

        X = []
        y = []

        for op in historial:
            try:
                # Obtener datos para el par y momento de la operación
                df = self.obtener_datos(op['par'])
                
                # Simular features que