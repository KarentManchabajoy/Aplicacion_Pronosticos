import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Configuración para limpiar la consola 
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

app = Flask(__name__)

# Colores para las fichas 
COLORES_BASE = ["#1abc9c", "#3498db", "#e67e22", "#9b59b6", "#f1c40f", "#e74c3c"]

# 1. FUNCIÓN DE MÉTRICAS 
def calcular_metricas(real, pred, n_inicio=3):
    y_true = np.array(real[n_inicio:], dtype=float)
    y_pred = np.array(pred[n_inicio:len(real)], dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return 0, 0, 0, 0

    error = y_pred - y_true
    error_abs = np.abs(error)
    
    # MAPE y MAPE' (evitando división por cero)
    ape = error_abs / np.where(y_true == 0, 1e-10, y_true)
    ape_p = error_abs / np.where(y_pred == 0, 1e-10, y_pred)
    
    error_cuadrado = error * error

    # Medidas de error finales
    MAPE = np.mean(ape)
    MAPE_prima = np.mean(ape_p)
    MSE = np.mean(error_cuadrado)
    RMSE = MSE**0.5 

    return round(MAPE, 4), round(MAPE_prima, 4), round(MSE, 2), round(RMSE, 2)


# 2. RUTA PRINCIPAL
@app.route("/", methods=["GET", "POST"])
def index():
    # Carga de datos 
    df = pd.read_csv("ventas_productos.csv", sep=";")
    productos_lista = list(df.columns[1:])
    fechas_raw = pd.to_datetime(df.iloc[:, 0])
    fechas_str = fechas_raw.dt.strftime("%Y-%m").tolist()
    ultima_fecha_dt = fechas_raw.max()
    
    # Valores por defecto
    fichas = []
    prod_sel = productos_lista[0]
    n, meses, metodo_sel = 3, 6, "todos"
    mostrar_ganador = False

    if request.method == "POST":
        prod_sel = request.form.get("producto")
        n = int(request.form.get("n", 3))
        meses = int(request.form.get("meses", 6))
        metodo_sel = request.form.get("metodo", "todos")
        mostrar_ganador = (metodo_sel == "todos")

    # Generar etiquetas del eje X (Pasado + Futuro)
    labels_eje_x = fechas_str.copy()
    for i in range(1, meses + 1):
        futuro = ultima_fecha_dt + relativedelta(months=i)
        labels_eje_x.append(futuro.strftime("%Y-%m"))

    # Procesar cada producto
    for i, prod in enumerate(productos_lista):
        serie = pd.to_numeric(df[prod], errors='coerce').fillna(0)
        reales = serie.tolist()
        
        # --- CÁLCULOS DE MÉTODOS ---
        
        # 1. Promedio Móvil
        p_prom = serie.rolling(window=n).mean().shift(1).bfill().tolist()
        p_prom += [serie.iloc[-n:].mean()] * meses
        
        # 2. SES
        try:
            model_ses = SimpleExpSmoothing(reales, initialization_method="estimated").fit()
            p_ses = list(model_ses.fittedvalues) + list(model_ses.forecast(meses))
        except:
            p_ses = [serie.mean()] * (len(reales) + meses)
        
        # 3. Prophet
        df_p = pd.DataFrame({'ds': fechas_raw, 'y': serie})
        try:
            m_p = Prophet(yearly_seasonality='auto', weekly_seasonality=False, daily_seasonality=False)
            m_p.fit(df_p)
            future = m_p.make_future_dataframe(periods=meses, freq='MS')
            p_prop = m_p.predict(future)['yhat'].tolist()
        except:
            p_prop = [serie.mean()] * (len(reales) + meses)

        # LÓGICA DE FILTRADO 
        
        # Lista con todos los cálculos posibles
        todos_los_metodos = [
            {"id": "promedio", "nombre": "Promedio Móvil", "m": calcular_metricas(reales, p_prom, n), "data": p_prom, "color": "#3498db"},
            {"id": "suavizacion", "nombre": "Suavización Exponencial", "m": calcular_metricas(reales, p_ses, n), "data": p_ses, "color": "#e67e22"},
            {"id": "prophet", "nombre": "Prophet", "m": calcular_metricas(reales, p_prop, n), "data": p_prop, "color": "#9b59b6"}
        ]

        mejor_abs = min(todos_los_metodos, key=lambda x: x["m"][3])

        # Filtramos la lista que se enviará al HTML según la elección del usuario
        if metodo_sel == "todos":
            met_list = todos_los_metodos
            seleccionado = mejor_abs
        else:
            # Filtramos para que solo quede el ID seleccionado
            met_list = [m for m in todos_los_metodos if m["id"] == metodo_sel]
            seleccionado = met_list[0]

        fichas.append({
            "nombre": prod, 
            "color_base": COLORES_BASE[i % len(COLORES_BASE)],
            "mejor": seleccionado, 
            "mejor_nombre": mejor_abs["nombre"], # Siempre muestra el nombre del mejor real
            "metodos": met_list, # Solo contiene los métodos filtrados
            "reales": reales, 
            "rmse_ganador": seleccionado["m"][3]
        })

    # Mejor método (Solo si se comparan todos)
    if mostrar_ganador and fichas:
        mejor_ficha = min(fichas, key=lambda x: x["rmse_ganador"])
        for f in fichas:
            f["es_el_mejor_global"] = (f["nombre"] == mejor_ficha["nombre"])

    return render_template(
        "pronosticos.html",
        productos=productos_lista,
        fichas=fichas,
        n=n,
        meses=meses,
        metodo_sel=metodo_sel,
        labels=labels_eje_x, 
        prod_sel=prod_sel,
        ultima_fecha=fechas_str[-1],
        mostrar_ganador=mostrar_ganador
    )

if __name__ == "__main__":
    app.run(debug=True)