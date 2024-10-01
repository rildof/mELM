import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import mode
import time
import copy
from sklearn.datasets import make_classification
import re
import numpy as np
import random


points_size = 4
samples_size = 10
text_size = 30


class Plotter:

    def __init__(self):
        pass

    def plotar(author, pasta, ActivationFunction, NumberofHiddenNeurons, 
            entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc, 
            InputWeight=None, InputWeightClass=None):

        # Create scatter plot for xx1, yy1 (blue 'x') and xx2, yy2 (red 'o')
        fig = go.Figure()
        
        # Plot testing data points
        fig.add_trace(go.Scatter(x=xx1, y=yy1, mode='markers',
                                marker=dict(size=points_size, color='blue', symbol='x'),
                                name='Class 1', opacity=0.6))
        
        fig.add_trace(go.Scatter(x=xx2, y=yy2, mode='markers',
                                marker=dict(size=points_size, color='red', symbol='circle'),
                                name='Class 2',opacity=0.6))
        
        # Handling InputWeight and InputWeightClass similar to plotar_Rildo
        if InputWeight is not None and InputWeightClass is not None:
            # Find the indices and filter benign and malignant entries
            indice_linhas_1 = np.where(InputWeightClass == 1)[0]
            x3 = InputWeight[indice_linhas_1, 0]
            y3 = InputWeight[indice_linhas_1, 1]

            indice_linhas_2 = np.where(InputWeightClass == 2)[0]
            x4 = InputWeight[indice_linhas_2, 0]
            y4 = InputWeight[indice_linhas_2, 1]

            # Remove duplicated lines of benign and malignant entries
            entrada_benigno_filtered = entrada_benigno[~np.isin(entrada_benigno, np.vstack((x3, y3)).T).all(axis=1)]
            entrada_maligno_filtered = entrada_maligno[~np.isin(entrada_maligno, np.vstack((x4, y4)).T).all(axis=1)]

            x1 = entrada_benigno_filtered[:, 0]
            y1 = entrada_benigno_filtered[:, 1]

            x2 = entrada_maligno_filtered[:, 0]
            y2 = entrada_maligno_filtered[:, 1]
        else:
            # If no InputWeight or InputWeightClass, just plot the data
            x1 = entrada_benigno[:, 0]
            y1 = entrada_benigno[:, 1]

            x2 = entrada_maligno[:, 0]
            y2 = entrada_maligno[:, 1]

        # Plot training data points for benign and malignant
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers',
                                marker=dict(size=samples_size, color='blue', symbol='x', line=dict(width=1.25)),
                                name='Benigno'))

        fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers',
                                marker=dict(size=samples_size, color='red', symbol='circle', line=dict(width=1.25)),
                                name='Maligno'))

        # Add text over points if InputWeight is provided
        if InputWeight is not None and InputWeightClass is not None:
            for i in range(len(x3)):
                fig.add_trace(go.Scatter(
                    x=[x3[i]], y=[y3[i]],
                    mode='text',
                    text=[str(indice_linhas_1[i])],
                    textposition='middle center',
                    textfont=dict(size=text_size, color='blue'),
                    showlegend=False
                ))

            for i in range(len(x4)):
                fig.add_trace(go.Scatter(
                    x=[x4[i]], y=[y4[i]],
                    mode='text',
                    text=[str(indice_linhas_2[i])],
                    textposition='middle center',
                    textfont=dict(size=text_size, color='red'),
                    showlegend=False
                ))

        # Adjust axis appearance
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0), 
            autosize=True,
            title=f'{author}|{ActivationFunction}|{NumberofHiddenNeurons} Hidden Neurons|Accuracy: {acc:.2f}%',
            title_x=0.5,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            showlegend=False
        )

        # Display the figure
        fig.show()
        if not os.path.exists("ELM_XAI"):
            os.mkdir("images")
        if not os.path.exists(f"ELM_XAI/{pasta}"):
            os.mkdir(f"ELM_XAI/{pasta}")
        #fig.write_image(f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png')
        #pio.write_image(fig, f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png',scale=2, width=864, height=720)

        
    def plotar_Rildo(author, pasta, ActivationFunction, NumberofHiddenNeurons,
                    InputWeight, InputWeightClass, 
                    entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc):
        # points_size = 5
        # samples_size = 10
        # text_size = 30
        # Criação da figura
        fig = go.Figure()

        # Scatter plot para xx1, yy1 e xx2, yy2
        fig.add_trace(go.Scatter(
            x=xx1, y=yy1,
            mode='markers',
            marker=dict(symbol='x', size=points_size, color='blue'),
            name='Class 1',
            opacity=0.6
        ))

        fig.add_trace(go.Scatter(
            x=xx2, y=yy2,
            mode='markers',
            marker=dict(symbol='circle', size=points_size, color='red'),
            name='Class 2',
            opacity=0.6
        ))

        # Encontrar os índices e filtrar entradas benignas e malignas
        indice_linhas_1 = np.where(InputWeightClass == 1)[0]
        x3 = InputWeight[indice_linhas_1, 0]
        y3 = InputWeight[indice_linhas_1, 1]

        indice_linhas_2 = np.where(InputWeightClass == 2)[0]
        x4 = InputWeight[indice_linhas_2, 0]
        y4 = InputWeight[indice_linhas_2, 1]

        # Remover as linhas duplicadas de entrada_benigno e entrada_maligno
        entrada_benigno_filtered = entrada_benigno[~np.isin(entrada_benigno, np.vstack((x3, y3)).T).all(axis=1)]
        entrada_maligno_filtered = entrada_maligno[~np.isin(entrada_maligno, np.vstack((x4, y4)).T).all(axis=1)]

        x1 = entrada_benigno_filtered[:, 0]
        y1 = entrada_benigno_filtered[:, 1]

        x2 = entrada_maligno_filtered[:, 0]
        y2 = entrada_maligno_filtered[:, 1]

        # Scatter plot para entradas benignas e malignas
        fig.add_trace(go.Scatter(
            x=x1, y=y1,
            mode='markers',
            marker=dict(symbol='x', size=samples_size, color='blue', line=dict(width=1.25)),
            name='Benigno'
        ))

        fig.add_trace(go.Scatter(
            x=x2, y=y2,
            mode='markers',
            marker=dict(symbol='circle', size=samples_size, color='red', line=dict(width=1.25)),
            name='Maligno'
        ))

        # Adicionar textos sobre os pontos
        for i in range(len(x3)):
            fig.add_trace(go.Scatter(
                x=[x3[i]], y=[y3[i]],
                mode='text',
                text=[str(indice_linhas_1[i])],
                textposition='middle center',
                textfont=dict(size=text_size, color='blue'),
                showlegend=False
            ))

        for i in range(len(x4)):
            fig.add_trace(go.Scatter(
                x=[x4[i]], y=[y4[i]],
                mode='text',
                text=[str(indice_linhas_2[i])],
                textposition='middle center',
                textfont=dict(size=text_size, color='red'),
                showlegend=False
            ))

        # Ajuste da aparência dos eixos
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Atualizar o layout com título e outros ajustes
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0), 
            autosize=True,
            title=f'{author}|{ActivationFunction}|{NumberofHiddenNeurons} Hidden Neurons|Accuracy: {acc:.2f}%',  # Adicionar título com precisão,
            title_x=0.5,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            showlegend=False
        )

        # Exibir a figura
        #fig.show()
        if not os.path.exists("ELM_XAI"):
            os.mkdir("images")
        if not os.path.exists(f"ELM_XAI/{pasta}"):
            os.mkdir(f"ELM_XAI/{pasta}")
        #fig.write_image(f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png')
        pio.write_image(fig, f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png',scale=2, width=864, height=720)
