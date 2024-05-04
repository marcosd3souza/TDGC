import utils
from data_loader import DataLoader, Dataset
from model import Model

data, G, y, n_clusters = DataLoader(Dataset.CORA).load()

print(f'Dataset: {data}:')
print('======================')
print(f'Number of graphs: {len(data)}')
print(f'Data shape: {data.x.shape}')
print(f'Number of features: {data.num_features}')
# print(f'Number of classes: {data.num_classes}')

model = Model(data)
embedding, y_pred = model.train()
utils.evaluate(embedding, y, y_pred)
