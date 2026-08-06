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


def remove_onlysenders(dataset: pd.DataFrame):
  '''
    Somente aplicável em datasets provenientes do SAML-D. Reduz iterativamente o dataset removendo arestas cujo sender nunca foi receiver, até todos terem sido receivers.
  '''
  df = dataset
  iteration = 0
  while True:
    accounts = pd.concat([df['Sender_account'], df['Receiver_account']]).unique()
    accounts_set = set(accounts)

    senders = set(df['Sender_account'])
    receivers = set(df['Receiver_account'])

    #print(f'Quantos accounts pssuem arestas bidirecionais?   {len(senders.intersection(receivers))} ({len(senders.intersection(receivers))/len(accounts_set)*100}%)')
    #print(f'Quantos accounts foram só senders?              {len(senders - receivers)} ({len(senders - receivers)/len(accounts_set)*100}%)')
    #print(f'Quantos accounts foram só receivers?            {len(receivers - senders)} ({len(receivers - senders)/len(accounts_set)*100}%)')

    bad_transactions = df[df['Sender_account'].isin(senders - receivers) == True]

    #print('Iteração', iteration)
    #print(f'Total de transações no dataset:   {df.size}')
    #print(f'Quantidade de "transações ruins": {bad_transactions.size} ({bad_transactions.size/df.size * 100})')

    if(bad_transactions.size == 0):
      very_good_transactions = df
      break
    else:
      iteration += 1

    # Quantos receiver accounts são só receivers
    #df[df['Receiver_account'].isin(receivers - senders) == True].size/df.size * 100
    #print('Quantidade de receiver que são só receivers:', df[df['Receiver_account'].isin(receivers - senders) == True].size/df.size * 100)

    good_transactions = df[df['Sender_account'].isin(senders - receivers) == False]

    df = good_transactions
    #print()
    #print('='*50)
    #print()

  print('Última iteração:', iteration)
  print('Quantidade de nós restantes:', very_good_transactions['Receiver_account'].unique().size)
  print('Porcentagem de transações removidas:', very_good_transactions.size/dataset.size)
  return very_good_transactions


