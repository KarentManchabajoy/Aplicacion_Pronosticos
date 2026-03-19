from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

def modelo_pronostico(serie, n):

    # Limpieza
    serie = pd.to_numeric(serie, errors='coerce').dropna().reset_index(drop=True)

    # Pronóstico (promedio móvil)
    pronostico = serie.rolling(window=n).mean().shift(1)

    # Errores
    error = pronostico - serie
    abs_error = error.abs()

    
    # MAPE
    ape = abs_error / serie.replace(0, pd.NA)
    mape = round(float(ape.dropna().mean() * 100), 2)

    # MAPE' 
    ape_p = abs_error / pronostico.replace(0, pd.NA)
    mape_p = round(float(ape_p.dropna().mean() * 100), 2)

    # MSE y RMSE
    error_cuadrado = error ** 2
    mse = round(float(error_cuadrado.dropna().mean()), 2)
    rmse = round(float(mse ** 0.5), 2)

    # Siguiente periodo
    siguiente = round(float(serie.iloc[-n:].mean()), 2)

   
    tabla = []

    for i in range(len(serie)):
        real = round(float(serie.iloc[i]), 2)

        pron = round(float(pronostico.iloc[i]), 2) if not pd.isna(pronostico.iloc[i]) else None
        err  = round(pron - real, 2) if pron is not None else None

        tabla.append({
            "periodo": i + 1,
            "real": real,
            "pronostico": pron,
            "error": err,
            "abs_error": round(abs(err), 2) if err is not None else None,
            "error_cuadrado": round(err**2, 2) if err is not None else None
        })

    return {
        "tabla": tabla,
        "siguiente": siguiente,
        "mape": mape,
        "mape_p": mape_p,
        "mse": mse,
        "rmse": rmse
    }



@app.route("/", methods=["GET", "POST"])
def index():

    # Lectura de datos
    df = pd.read_csv("ventas_productos.csv", sep=";")

    productos = df.columns[1:]
    periodos = df.iloc[:, 0].tolist()

    resultados = None
    producto_seleccionado = None
    n = 3

    if request.method == "POST":

        producto_seleccionado = request.form.get("producto")
        n = int(request.form.get("n"))

        resultados = modelo_pronostico(df[producto_seleccionado], n)

    return render_template(
        "pronosticos.html",
        productos=productos,
        periodos=periodos,
        resultados=resultados,
        producto_seleccionado=producto_seleccionado,
        n=n
    )





if __name__ == "__main__":
    app.run(debug=True)