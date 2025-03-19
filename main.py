from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

mlp_map = {
    'teste2.npy': [
        MLPRegressor(
            hidden_layer_sizes=(100, 75, 40),
            max_iter=1000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
       MLPRegressor(
            hidden_layer_sizes=(200, 100, 50),
            max_iter=1000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
        MLPRegressor(
            hidden_layer_sizes=(60, 40, 20),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        )
    ],
    'teste3.npy': [
        MLPRegressor(
            hidden_layer_sizes=(200, 100, 70),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
       MLPRegressor(
            hidden_layer_sizes=(200, 175, 150),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
        MLPRegressor(
            hidden_layer_sizes=(200, 150, 100),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        )
    ],
    'teste4.npy': [
        MLPRegressor(
            hidden_layer_sizes=(375, 350),
            max_iter=1500,
            activation='relu', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
       MLPRegressor(
            hidden_layer_sizes=(200, 175, 150),
            max_iter=1000,
            activation='relu', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
        MLPRegressor(
            hidden_layer_sizes=(400, 300, 200, 100),
            max_iter=500,
            activation='relu', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        )
    ],
    'teste5.npy': [
        MLPRegressor(
            hidden_layer_sizes=(300, 260, 210, 150, 90),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
       MLPRegressor(
            hidden_layer_sizes=(250, 200, 150, 100, 50),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        ), 
        MLPRegressor(
            hidden_layer_sizes=(300, 240, 180, 120, 60),
            max_iter=10000,
            activation='tanh', 
            solver='adam',
            learning_rate = 'adaptive',
            n_iter_no_change=50
        )
    ],
}

for i in range(2, 6):
    filename = f"teste{i}.npy"
    print(f'Carregando Arquivo {filename}')

    file = np.load(filename)

    x = file[0]
    y = np.ravel(file[1])

    architectures = mlp_map[filename]

    best_loss_curve = None
    best_y_est = None
    best_loss = 10**10
    best_arch = 1
    best_mean = 10**10

    for i, regr in enumerate(architectures):
        print(f'ARQUITETURA {i + 1}')
        total_sum = 0
        n = 10
        losses = []

        for j in range(n):
            print(f'\nSimulação {j + 1}:')
            print('>>> Treinando')
            regr = regr.fit(x,y)

            print('>>> Estimando\n')
            y_est = regr.predict(x)

            losses.append(regr.best_loss_)
            total_sum += regr.best_loss_

        mean = total_sum / n

        std_deviation_sum = 0

        for loss in losses:
            std_deviation_sum += (loss - mean)**2
        
        std_deviation = sqrt(std_deviation_sum / n)

        min_loss = min(losses)

        print(f'Dados para arquitetura {i + 1}:\nMédia: {mean}\nDesvio padrão: {std_deviation}\nMelhor perda: {min_loss}\n')

        if mean < best_mean:
            best_loss = min_loss
            best_mean = mean
            best_loss_curve = regr.loss_curve_
            best_y_est = y_est
            best_arch = i + 1

    fig = plt.figure(figsize=[14,7])
    fig.suptitle(f"Arquitetura escolhida para {filename}: arq. nº {best_arch} (Média: {best_mean} <> Melhor perda: {best_loss})")
    plt.subplot(1,3,1)
    plt.title('Função original')
    plt.plot(x,y)
    plt.subplot(1,3,2)
    plt.title('Função de perda')
    plt.plot(best_loss_curve)
    plt.subplot(1,3,3)
    plt.title('Função original X estimada')
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x, best_y_est,linewidth=2)
    plt.show()