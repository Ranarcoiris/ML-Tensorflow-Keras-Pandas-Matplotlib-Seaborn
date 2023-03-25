from tensorflow import keras
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers 


#DESCARGAR EL DATASET 
def load_data():

    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    #print(dataset.isna().sum())

    return (dataset, raw_dataset)

def clean_data(data):

    data = data.dropna() #Elimina los valores no deseados

    origin = dataset.pop('Origin') #Eliminamos la columna de origin porque son destinos y no lo queremos en una catergoria(1,2,3)
    #Esa variable de origin tiene los datos del pop
    
    #Aca creamos las columnas que vamos a ingresar 
    dataset['USA'] = (origin == 1)*1.0 #Si sale true == 1, por lo que seria 1*1
    dataset['Europe'] = (origin == 2)*1.0 #Si sale false == 0, esto llevaria a que es 0+1
    dataset['Japan'] = (origin == 3)*1.0
    dataset.tail()

    return data


def describe_data(train_dataset):
    #sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    #plt.show() #imprime las graficas

    #El dataset se describio con la libreria sns-> que ayuda a visualizar el dataset con matplotlib
    

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    return train_stats

def build_model(train_dataset):
  estructura_neuronal = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]), #Se genera la primera capa de red neuronal
    layers.Dense(64, activation='relu'), #Segunda capa
    layers.Dense(1) #Tercera capa aca solo le ponemos uno porque queremos solo un valor 
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001) #se utiliza el optimizador disminuye el tiempo de entrenamiento 
                                                 #Ayuda tambien para encontrar los valores optimos para el modelo


  estructura_neuronal.compile(loss='mse',               #Compilamos el modelo con la funcion de perdida que queremos optimizar
                                          #en este caso queremos hacerlo con el Min Square Error (mse) 
                optimizer=optimizer,
                metrics=['mae', 'mse'])   #se definen las metricas del [entrenamiento , test]-> Min Absolute Error (mae)
  return estructura_neuronal


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


    plot_history(history)

if __name__== "__main__":
    pd.set_option("display.max_columns",10)
    pd.options.mode.chained_assignment = None
    #Descarga el dataset------------------------------------------------------------------------------
    dataset, raw_dataset= load_data()

    #Clean the data-----------------------------------------------------------------------------------
    dataset = clean_data(dataset)

    # Separar los datos en entrenamento y test---------------------------------------------------------
    train_dataset = dataset.sample(frac=0.8,random_state=0) #aca lo que logra es hacer el test con 80% de los datos aleatoriamente
    test_dataset = dataset.drop(train_dataset.index) #aca elimina los datos de prueba del data set inicial
                                                    # ya que seria irreal que se pruebe con un modelo que ya se probo
                                                    # el index es porque saca las pociciones del data frame de prueba
                                                    # y los elimina del original para meterlo en la variable de test
    # Vizualización de datos----------------------------------------------------------------------------
    train_stats=describe_data(train_dataset)

    #Separamos los atributos de las etiquetas------------------------------------------------------------
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    #Normalización---------------------------------------------------------------------------------------
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std'] #normaliza los datos
    
    normed_train_data = norm(train_dataset) #se crean las variables normalizadas tanto de el Train como del test    
    normed_test_data = norm(test_dataset)

    #Construcción del modelo-----------------------------------------------------------------------------
    modelo_neuronas = build_model(train_dataset)
    #print(modelo_neuronas.summary()) Deja ver la cantidad de neuras conectadas y sus parametros con los bias


    #example_batch = normed_train_data[:10]
    #example_result = modelo_neuronas.predict(example_batch)
    #print(example_result)

    #Entrenar el modelo----------------------------------------------------------------------------------
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 1000

    early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


    history = modelo_neuronas.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[early_stopping ,  PrintDot()]) #Este fit se utiliza para entrenar el modelo y hacer que disminuya cada vez mas el error

    #Visualizacion de progreso--------------------------------------------------------------------------
    #plot_history(history)


    loss, mae, mse = modelo_neuronas.evaluate(normed_test_data, test_labels, verbose=2) #Aca lo que se queria era evalauar son de acuerdo a
                                                                                        #Las variables de entrada loss, mae, mse son las que
                                                                                        #asigna el formato de 
    print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(mae))

    test_predictions = modelo_neuronas.predict(normed_test_data).flatten() #Aca lo que se hace con el predict es usar ese modelo de neuronas 
                                                                           #Para aplicarlo en los datos de test
                                                                            

    """plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    """
    #plt.show() #En esta grafica lo que se quiere es mostrar que el modelo se adecua mucho a los datos 

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()