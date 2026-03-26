from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet

app = Flask(__name__)

colores = {
    "Producto A": "#1abc9c",
    "Producto B": "#3498db",
    "Producto C": "#e67e22",
    "Producto D": "#9b59b6",
    "Producto E": "#e74c3c"
}

# PROMEDIO MÓVIL

def modelo_promedio(serie, n, periodos):
    serie = pd.to_numeric(serie, errors='coerce').dropna().reset_index(drop=True)
    pronostico = serie.rolling(window=n).mean().shift(1)
    futuro = [serie.iloc[-n:].mean()] * periodos
    return list(pronostico) + futuro



# SES - statsmodels 
def modelo_ses(serie, periodos):
    serie = pd.to_numeric(serie, errors='coerce').dropna()
    model = SimpleExpSmoothing(serie)
    fit = model.fit()
    return list(fit.fittedvalues) + list(fit.forecast(periodos))


# PROPHET

def modelo_prophet(df, columna, periodos):
    data = pd.DataFrame({
        "ds": pd.to_datetime(df.iloc[:, 0]),
        "y": pd.to_numeric(df[columna])
    }).dropna()

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(data)

    future = model.make_future_dataframe(periods=periodos, freq='MS')
    forecast = model.predict(future)

    return forecast["yhat"].tolist()



# MÉTRICAS

def calcular_metricas(real, pred):

    real = pd.to_numeric(pd.Series(real), errors='coerce')
    pred = pd.to_numeric(pd.Series(pred), errors='coerce')

    min_len = min(len(real), len(pred))
    real = real[-min_len:]
    pred = pred[-min_len:]

    df = pd.DataFrame({
        "real": real.values,
        "pred": pred.values
    })

    df["error"] = df["pred"] - df["real"]
    df["abs_error"] = df["error"].abs()

    df["ape"] = df["abs_error"] / df["real"].replace(0, pd.NA)
    df["ape_p"] = df["abs_error"] / df["pred"].replace(0, pd.NA)

    df["error_cuadrado"] = df["error"] ** 2

    mape = float(df["ape"].dropna().mean() * 100)
    mape_p = float(df["ape_p"].dropna().mean() * 100)
    mse = float(df["error_cuadrado"].dropna().mean())
    rmse = float(np.sqrt(mse))

    return mape, mape_p, mse, rmse


# RUTA PRINCIPAL

@app.route("/", methods=["GET", "POST"])
def index():

    df = pd.read_csv("ventas_productos.csv", sep=";")
    productos = df.columns[1:]

    ultima_fecha_dt = pd.to_datetime(df.iloc[:, 0]).max()
    ultima_fecha = ultima_fecha_dt.strftime("%Y-%m")

    resultados = None
    comparacion = []
    producto = None
    n = 3
    meses = 6
    metodo = "todos"

    if request.method == "POST":

        producto = request.form.get("producto")
        n = int(request.form.get("n"))
        meses = int(request.form.get("meses"))
        metodo = request.form.get("metodo")

        serie = df[producto]
        reales = pd.to_numeric(serie, errors='coerce').dropna().tolist()

        p_prom, p_ses, p_prop = [], [], []

        if metodo in ["todos", "promedio"]:
            p_prom = modelo_promedio(serie, n, meses)

        if metodo in ["todos", "ses"]:
            p_ses = modelo_ses(serie, meses)

        if metodo in ["todos", "prophet"]:
            p_prop = modelo_prophet(df, producto, meses)

        comparacion = []

        if p_prom:
            mape, mape_p, mse, rmse = calcular_metricas(reales, p_prom)
            comparacion.append({"producto": producto, "modelo": "Promedio",
                                "pronostico": round(p_prom[-1],2),
                                "mape": f"{round(mape,2)}%",
                                "mape_p": f"{round(mape_p,2)}%",
                                "mse": round(mse,2), "rmse": round(rmse,2)})

        if p_ses:
            mape, mape_p, mse, rmse = calcular_metricas(reales, p_ses)
            comparacion.append({"producto": producto, "modelo": "SES",
                                "pronostico": round(p_ses[-1],2),
                                "mape": f"{round(mape,2)}%",
                                "mape_p": f"{round(mape_p,2)}%",
                                "mse": round(mse,2), "rmse": round(rmse,2)})

        if p_prop:
            mape, mape_p, mse, rmse = calcular_metricas(reales, p_prop)
            comparacion.append({"producto": producto, "modelo": "Prophet",
                                "pronostico": round(p_prop[-1],2),
                                "mape": f"{round(mape,2)}%",
                                "mape_p": f"{round(mape_p,2)}%",
                                "mse": round(mse,2), "rmse": round(rmse,2)})

        mejor_modelo = min(comparacion, key=lambda x: x["rmse"]) if comparacion else None

        resultados = {
            "reales": reales,
            "promedio": p_prom,
            "ses": p_ses,
            "prophet": p_prop
        }

        fechas = pd.to_datetime(df.iloc[:, 0]).tolist()
        for i in range(meses):
            fechas.append(fechas[-1] + pd.DateOffset(months=1))

        labels = [f.strftime("%Y-%m") for f in fechas]

        return render_template("pronosticos.html",
                               productos=productos,
                               periodos=labels,
                               resultados=resultados,
                               comparacion=comparacion,
                               colores=colores,
                               producto=producto,
                               mejor_modelo=mejor_modelo,
                               n=n,
                               meses=meses,
                               metodo=metodo,
                               ultima_fecha=ultima_fecha)

    return render_template("pronosticos.html",
                           productos=productos,
                           periodos=[],
                           resultados=None,
                           comparacion=[],
                           colores=colores,
                           producto=None,
                           n=n,
                           meses=meses,
                           metodo=metodo,
                           ultima_fecha=ultima_fecha)







if __name__ == "__main__":
    app.run(debug=True)