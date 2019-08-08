import numpy as np
from resnet1d import resnet_34
from Generator import DataGenerator

# Parameters
params = {'dim': (7201,1),
          'batch_size': 512,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
nb_classes = 2
nb_epoch = 50
# Datasets
ID_len = 15000000
val_percentage = 20
partition = {'train':[],'validation':[]}
labels = {}
y_train = np.random.randint(2, size=ID_len)
for item in range(ID_len):
    if item%val_percentage == 0:
        partition['validation'].append(item)
        labels[item] = y_train[item]
    else:
        partition['train'].append(item)
        labels[item] = y_train[item]

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = resnet_34(num_classes=nb_classes)
# Train model on dataset

model.fit_generator(generator=training_generator, validation_data=validation_generator,
              shuffle=True,
              verbose=2,
              epochs=nb_epoch,
              class_weight='auto')