'''
Copyright (C) 2020 Federico Sassi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
from datetime import datetime
import pandas as pd
import numpy as np

import pylab as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

# Script quick & dirty per valutare i posti in rianimazione, ovvero l'unica cosa che conta!
# Semplici modelli di regressione esponenziale e quadratico per farci una idea di dove possiamo andare
# Dati scaricati da https://github.com/pcm-dpc/COVID-19

# E' possibile modificare lo script per predire tutte le altre variabili
# Se si sceglie più di una colonna viene creato un unico plot
col_to_plot = ['terapia_intensiva', 'totale_attualmente_positivi']
col_to_plot = ['ricoverati_con_sintomi', 'terapia_intensiva', 'totale_ospedalizzati', 
             'isolamento_domiciliare', 'totale_attualmente_positivi', 'nuovi_attualmente_positivi', 'dimessi_guariti', 
             'deceduti', 'totale_casi', 'tamponi']
# col_to_plot = ['totale_attualmente_positivi'] #, 'ricoverati_con_sintomi']
# col_to_plot = ['totale_attualmente_positivi', 'nuovi_attualmente_positivi']
# col_to_plot = ['nuovi_attualmente_positivi']

FUTURE_DAYS = 7
DIR_DATI_NAZ = 'dati-andamento-nazionale'
FNAME_DATI_NAZ = 'dpc-covid19-ita-andamento-nazionale.csv'
PLOT_WITHOUT_PREDICTION = False


# ---------------------------------
# KEYS_NAZ = ['data', 'stato', 'ricoverati_con_sintomi', 'terapia_intensiva', 'totale_ospedalizzati', 
#             'isolamento_domiciliare', 'totale_attualmente_positivi', 'nuovi_attualmente_positivi', 'dimessi_guariti', 
#             'deceduti', 'totale_casi', 'tamponi']

# LOAD DATA
data_naz = pd.read_csv(os.path.join('dati-andamento-nazionale', FNAME_DATI_NAZ))
data_naz['data'] = pd.to_datetime(data_naz['data'])

for c in col_to_plot:
	ex_date = pd.to_datetime('2020-03-10 18:00:00') 
	idx = data_naz.index[data_naz['data'] == ex_date]
	data_naz[c][idx] =  (data_naz[c][idx+1].values + data_naz[c][idx-1].values) / 2.
	
	
	plt.figure()
	ax = plt.axes()
	plt.title(c + '\nInterpolazione ' + str(FUTURE_DAYS) + ' giorni dati COVID-19 Italia \n' + \
			  'Ultimi dati: ' + str(data_naz['data'].iloc[-1]) )

	X = np.arange(len(data_naz)).reshape(-1,1)
	X_FUTURE = np.arange(len(data_naz) + FUTURE_DAYS).reshape(-1,1)

	y = data_naz[c].values.reshape(-1,1)

	# QUADRATIC
	poly_reg = PolynomialFeatures(degree=2)
	X_poly = poly_reg.fit_transform(X)
	pol_reg = LinearRegression()
	pol_reg.fit(X_poly, y)
	plt.plot(X_FUTURE, pol_reg.predict(poly_reg.fit_transform(X_FUTURE)), 'r--')

	# EXPONENTIAL
	transformer = FunctionTransformer(np.log, validate=True)
	y_log = transformer.fit_transform(y)  
	model = LinearRegression().fit(X, y_log)
	y_fit = model.predict(X_FUTURE)
	plt.plot(X_FUTURE, np.exp(y_fit), "k--")

	# plot original data
	plt.scatter(X, y, label=c, s=72)

	date_future = pd.date_range(start=data_naz['data'].iloc[0] , periods=len(data_naz) + FUTURE_DAYS, freq='24H')
	date_future_str = [a.strftime('%d')  for a in date_future]
	X_future = np.arange(len(data_naz) + FUTURE_DAYS).reshape(-1,1)

	plt.xticks(X_future, date_future_str)
	plt.xlabel('Febbraio | Marzo')

	
	last_value = data_naz[c].iloc[-1]
	plt.axhline(last_value, color='green')
	plt.axhline(last_value*2, color='orange')

	# add terapia intensiva hlines
	if c == 'terapia_intensiva':
		# https://www.agi.it/fact-checking/news/2020-03-06/coronavirus-posti-letto-ospedali-7343251/
		plt.axhline(5090, color='red') 
		plt.yticks(list(plt.yticks()[0]) + [last_value, last_value * 2])

	# add legend
	handles, labels = ax.get_legend_handles_labels()
	handles.append(Line2D([0], [0], color='r', linewidth=3, linestyle='--'))
	labels.append('Interpolazione quadratica')

	handles.append(Line2D([0], [0], color='k', linewidth=3, linestyle='--'))
	labels.append('Interpolazione esponenziale')

	handles.append(Line2D([0], [0], color='g', linewidth=3, linestyle='-'))
	labels.append('Valore attuale')

	handles.append(Line2D([0], [0], color='orange', linewidth=3, linestyle='-'))
	labels.append('Raddoppio attuale')

	if c == 'terapia_intensiva':
		handles.append(Line2D([0], [0], color='r', linewidth=3, linestyle='-'))
		labels.append('Totale posti in rianimazione in Italia')

	plt.legend(handles=handles, labels=labels, loc='upper left')
	plt.grid()


# PLOT CURRENT STATUS
if PLOT_WITHOUT_PREDICTION:
    col_to_plot = ['terapia_intensiva', 'ricoverati_con_sintomi']
    plt.figure()
    plt.title('Dati COVID-19 Italia' + '\n' + str(data_naz['data'].iloc[-1]))

    for c in col_to_plot:
        plt.plot(data_naz['data'], data_naz[c], 'o-', label=c)
    plt.legend()
    plt.grid()

    # prettify labels
    ax = plt.axes()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
#--------------

plt.show()
