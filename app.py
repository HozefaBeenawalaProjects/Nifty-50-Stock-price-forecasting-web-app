import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet.plot import plot_plotly, plot_components_plotly

app = Flask(__name__)

def StockPriceForecasting(Symbol, Days):
    # Use fivethirtyeight plot style
    plt.style.use('fivethirtyeight')
    
    data = yf.download(tickers=Symbol, period='1y', interval='1d')
    data.reset_index(inplace=True)
    
    # Add two columns in the dataframe with values as Date and Adj Close
    data[['ds', 'y']] = data[['Date', 'Adj Close']]
    
    # Subset two columns from the data frame
    data = data[['ds', 'y']]
    
    # Initialize the Model
    model = Prophet()
    model.fit(data)
    
    # Create future dates of `Days` duration
    future_dates = model.make_future_dataframe(periods=Days)
    prediction = model.predict(future_dates)
    
    # Save plot as HTML file
    plot_file = os.path.join('static', 'plot.html')
    plot_plotly(model, prediction).write_html(os.path.join(app.root_path, plot_file))
    
    return plot_file


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('Symbol')
        days = int(request.form.get('Days'))
        plot_file = StockPriceForecasting(symbol, days)
        return redirect(url_for('plot', plot_file=plot_file))
    
    return render_template('index.html')


@app.route('/plot')
def plot():
    plot_file = request.args.get('plot_file', '')
    return render_template('plot.html', plot_file=plot_file)


if __name__ == '__main__':
    app.run()