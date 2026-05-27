# Imports
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, auc, mean_squared_error

def bizoia_resultados(n: int, embedding_size: int):
    # bizoia n resultados e dá a média dos trem
    media = {'accuracy': 0, 'precision': 0, 'recall': 0}
    for i in range(n):
        print(f'--- Resultado {i} ---')
        with open(f'node_classification/Resultados/predicoes/predict{i}.pkl', 'rb') as f:
            data = pickle.load(f)

        preds = data['preds']
        labels = data['labels']
        nodes = data['nodes']

        resultado = pd.DataFrame.from_dict({'preds': data['preds'], 'labels': data['labels']})
        
        with open(f'node_classification/Resultados/embeddings/embedding{i}.pkl', 'rb') as f:
            embed = pickle.load(f)
        
        s = 0
        for e in embed:
            s += len(e)

        print('Nós embeddados:', s)
        print('Nós labelados:', len(labels))
        print('Porcentagem de nós embeddados:', s/embedding_size, end='\n\n')

        print(resultado['preds'].value_counts(), end='\n\n')
        print(resultado['labels'].value_counts(), end='\n\n')

        accuracy = accuracy_score(resultado['labels'], resultado['preds'])
        precision = precision_score(resultado['labels'], resultado['preds'], zero_division=0)
        recall = recall_score(resultado['labels'], resultado['preds'])

        print(f'Accuracy {i}:', accuracy)
        print(f'Precision {i}:', precision)
        print(f'Recall {i}:', recall)
        media['accuracy'] += accuracy
        media['precision'] += precision
        media['recall'] += recall
        print()
        print()
    
    print('--- Média dos resultados ---')
    print('Accuracy:', media['accuracy']/n)
    print('Precision:', media['precision']/n)
    print('Recall:', media['recall']/n)
    print()

