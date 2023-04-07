#=====================================================================#
#    PROYECTO FINAL INTELIGENCIA ARTIFICIAL Y APRENDIZAJE AUTOMÁTICO  #                                             #
#                                                                     #
#    Nombre del Alumno             Matrícula                          #
#    Nantzin Gizel Nájera Gálvez  	A01451495                          #
#    Valeria Juárez Domínguez  	A01451531                          #
#    Ruth Rebeca Muñoz Chávez  	A01085863                          #
#                                                                     #
#=====================================================================#



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer   
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from pylab import rcParams
 
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
 
from collections import Counter


##################################################################
# Parte I: Partición, análisis y pre-procesamiento de los datos. #
##################################################################
data = pd.read_csv("/Users/nantzingizelnajeragalvez/Downloads/proyectoFinalInteligencia/SouthGermanCredittrain.csv", sep=",") 
print(data.head())
print(data.info())

#1. Realiza una partición de los datos en el conjunto de entrenamiento y el de prueba. 
# Los modelos se estarán entrenando con el método de validación cruzada, 
# así que no es necesario en este paso generar el conjunto de validación. 
# Deberás proponer los porcentajes en la partición que consideres más adecuada.

#separar datos 
tam_test=0.2
semilla=1234
training_data, testing_data = train_test_split(data, test_size=tam_test, random_state=semilla)

#definir las metricas a evaluar en la cross-validtion
metrics=['accuracy', 'recall']

#definir cantidad de iteraciones en cross-validation
cv_iterations=10

#2. Realiza un análisis descriptivo sobre el conjunto de datos y lleva a cabo las 
# técnicas de pre- procesamiento que consideres más adecuadas. 
# Justifica y documenta las decisiones tomadas. 
# Incluye un breve resumen con todas las transformaciones que hayas decidido aplicar.
print(training_data.describe())
cantGraficas=len(data.columns)
graficasPorFila=3
graficasPorColumna=3
graficasPorPagina=graficasPorFila*graficasPorColumna
cantPaginas=math.ceil(cantGraficas/graficasPorPagina)
sns.set(rc={'figure.figsize':(8,8)}) 
fig, axes = plt.subplots(graficasPorColumna, graficasPorFila)    # definimos una ventana de 3x3 nichos para incluir en cada uno de ellos un gráfico.
fig.suptitle('Analisis Descriptivo de los Datos', fontsize=16)
for i in range(0,cantPaginas):
    for k in range(0,graficasPorPagina):
        var=k+(graficasPorPagina*i)
        if(var>=cantGraficas):break
        plt.subplot(graficasPorColumna,graficasPorFila,k+1)     # los nichos para cada histograma se numeran iniciando en 1 y no en 0.
        plt.hist(training_data[training_data.columns[var]], bins=20)     # datatrain.columns nos devuelve una lista con los nombres de las columnas.
        plt.xlabel(training_data.columns[var])
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()

#Para identificar qué columnas es necesario transformar o ajustar en el dataset
#se validó primero la correlación de cada columna con la variable objetivo que
#es la columna credit_risk. A continuación se muestra el código para la matriz de
#correlación. 

#matriz de correlacion de datos de entrenamiento
corr=training_data.corr()
final_corr = corr.sort_values(by=['credit_risk'], ascending=False)
print("VARIABLES QUE SE CORRELACIONAN CON credit_risk EN ORDEN DESCENCENTE")
print(final_corr['credit_risk'])
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={'size': 'x-small'})
plt.show()

#Las columnas que mas se correlacionan con credit_risk son:
#status                    
#credit_history            
#savings                   
#other_installment_plans   
#employment_duration  
# 
# Entonces las columnas que vamos a usar para los modelos son
# status y credit_history, por lo tanto, son las que nos vamos
# a tomar la molestia de transformar.
# En este caso vamos a escalarlas para que sus valores oscilen
# entre 0 y 1, puesto que se ha comprobado que los modelos
# entregan mejores resultados cuando los datos están esclaados.     

X = data[['status','credit_history']].values
y = data['credit_risk'].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tam_test, random_state=semilla)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##################################################################
# Parte II: Modelos de aprendizaje automático: Modelo base.      #
##################################################################

#3. En este ejercicio debes usar el método de validación-cruzada (cross-validation) 
# para entrenar y evaluar de manera conjunta los siguientes métodos con sus parámetros predeterminados. 
# Es decir, en este ejercicio deja los hiperparámetros predeterminados de cada método sin modificar.
#     
#En dado caso, modifica solo aquellos que te permita entrenarlos sin errores o warnings, 
# como por ejemplo el máximo de iteraciones en algunos de ellos:
#
#   a. k-vecinos más cercanos (kNN), regresión logística (LR), máquina de vector soporte (SVM), 
#       red neuronal artificial (MLP), bosque aleatorio (RF). 
#       En cada uno deberás indicar los resultados de al menos las métricas “acuracy”, 
#       “recall”, “f1-score” y “f2-score”.


#KNN
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train, y_train)))
y_pred = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)
precision_knn = precision_score(y_test, y_pred)
recall_knn = recall_score(y_test, y_pred)
f1score_knn = f1_score(y_test, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the Best Value of k
k_values = [i for i in range (1,31)]
scores = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_validate(knn, X, y, cv=cv_iterations, scoring=metrics)
    scores.append(np.mean(score['test_accuracy']))
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()
#EL MEJOR VALOR PARA K ES 12, PORQUE QES EL VALOR MAS PEQUEÑO QUE TIENE MEJOR ACCURACY
#EL MEJOR DESEMPEÑO DE KNN PARA ESTE DATASET Y ALGORITMO ES CON K=12 
#CON ESE VALOR DE K SE ALCANZA UN ACCURACY DE 0.77
#Using Cross Validation to Get the general performance of the model
knn_scores = cross_validate(knn, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN', knn_scores)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores['test_accuracy'].min(), knn_scores['test_accuracy'].mean(), knn_scores['test_accuracy'].max()))



#LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train, y_train)
print("\n METRICAS EVALUACIÓN LOGISTIC REGRESSION \n")
print('Accuracy of Logistic Regression classifier on training set: {:.4f}'
     .format(model.score(X_train, y_train)))
y_pred = model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1score_lr = f1_score(y_test, y_pred)
print('Accuracy of Logistic Regression classifier on test set: {:.4f}'
     .format(accuracy_lr))
print('Precision of Logistic Regression classifier on test set: {:.4f}'
     .format(precision_lr))
print('Recall of Logistic Regression classifier on test set: {:.4f}'
     .format(recall_lr))
print('F1_Score of Logistic Regression classifier on test set: {:.4f}'
     .format(f1score_lr))
result_lr = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for Logistic Regression:")
print(result_lr)
color = 'black'
matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for Logistic Regression', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
lr_scores = cross_validate(model, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores fo Logistic Regression', lr_scores)
#lr_scores = pd.Series(lr_scores)
print('\n Min score for LR: {:.4f} \n Mean score for LR: {:.4f} \n Max score for LR: {:.4f}' .format(lr_scores['test_accuracy'].min(), lr_scores['test_accuracy'].mean(), lr_scores['test_accuracy'].max()))


#SVM
#Create a svm Classifier
svm_model = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
svm_model.fit(X_train, y_train)
print("\n METRICAS EVALUACIÓN SVM \n")
print('Accuracy of SVM classifier on training set: {:.4f}'
     .format(svm_model.score(X_train, y_train)))
#Predict the response for test dataset
y_pred = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
precision_svm = precision_score(y_test, y_pred)
recall_svm = recall_score(y_test, y_pred)
f1score_svm = f1_score(y_test, y_pred)
print('Accuracy of SVM classifier on test set: {:.4f}'
     .format(accuracy_svm))
print('Precision of SVM classifier on test set: {:.4f}'
     .format(precision_svm))
print('Recall of SVM classifier on test set: {:.4f}'
     .format(recall_svm))
print('F1_Score of SVM classifier on test set: {:.4f}'
     .format(f1score_svm))
result_svm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for SVM:")
print(result_svm)
color = 'black'
matrix = plot_confusion_matrix(svm_model, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for SVM', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
svm_scores = cross_validate(svm_model, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores fo SVM', svm_scores)
#svm_scores = pd.Series(svm_scores)
print('\n Min score for SVM: {:.4f} \n Mean score for SVM: {:.4f} \n Max score for SVM: {:.4f}' .format(svm_scores['test_accuracy'].min(), svm_scores['test_accuracy'].mean(), svm_scores['test_accuracy'].max()))


#MLP
#Initializing the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
#Fitting the training data to the network
mlp_classifier.fit(X_train, y_train)
print('Accuracy of MLP classifier on training set: {:.4f}'
     .format(mlp_classifier.score(X_train, y_train)))
#Predict the response for test dataset
y_pred = mlp_classifier.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred)
precision_mlp = precision_score(y_test, y_pred)
recall_mlp = recall_score(y_test, y_pred)
f1score_mlp = f1_score(y_test, y_pred)
print('Accuracy of MLP classifier on test set: {:.4f}'
     .format(accuracy_mlp))
print('Precision of MLP classifier on test set: {:.4f}'
     .format(precision_mlp))
print('Recall of MLP classifier on test set: {:.4f}'
     .format(recall_mlp))
print('F1_Score of MLP classifier on test set: {:.4f}'
     .format(f1score_mlp))
result_mlp = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for MLP:")
print(result_mlp)
color = 'black'
matrix = plot_confusion_matrix(mlp_classifier, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for MLP', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
mlp_scores = cross_validate(mlp_classifier, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores fo SVM', mlp_scores)
#mlp_scores = pd.Series(mlp_scores)
print('\n Min score for MLP: {:.4f} \n Mean score for MLP: {:.4f} \n Max score for MLP: {:.4f}' .format(mlp_scores['test_accuracy'].min(), mlp_scores['test_accuracy'].mean(), mlp_scores['test_accuracy'].max()))


#RF
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Accuracy of MLP classifier on training set: {:.4f}'
     .format(rf.score(X_train, y_train)))
#Predict the response for test dataset
y_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
precision_rf = precision_score(y_test, y_pred)
recall_rf = recall_score(y_test, y_pred)
f1score_rf = f1_score(y_test, y_pred)
print('Accuracy of RF classifier on test set: {:.4f}'
     .format(accuracy_rf))
print('Precision of RF classifier on test set: {:.4f}'
     .format(precision_rf))
print('Recall of RF classifier on test set: {:.4f}'
     .format(recall_rf))
print('F1_Score of RF classifier on test set: {:.4f}'
     .format(f1score_rf))
result_rf = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for RF:")
print(result_rf)
color = 'black'
matrix = plot_confusion_matrix(rf, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for RF', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
rf_scores = cross_validate(rf, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores fo RF', rf_scores)
#rf_scores = pd.Series(mlp_scores)
print('\n Min score for RF: {:.4f} \n Mean score for RF: {:.4f} \n Max score for RF: {:.4f}' .format(rf_scores['test_accuracy'].min(), rf_scores['test_accuracy'].mean(), rf_scores['test_accuracy'].max()))

#Hyperparameter Tuning for RF
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}
# Create a random forest classifier
rf = RandomForestClassifier()
# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)
# Create a variable for the best model
best_rf = rand_search.best_estimator_
# Print the best hyperparameters
print('Best hyperparameters for RF:',  rand_search.best_params_)


#b. Obtener un diagrama de cajas conjunto de todos los modelos con 
# los resultados de la métrica “accuracy” y otro diagrama de cajas conjunto de 
# todos los modelos con la métrica “recall”. 
# Incluye tus conclusiones al respecto, en particular indica cuáles consideras 
# son los mejores modelos obtenidos.

#ARMAR DATASET CON LOS SCORES PARA HACER BOXPLOT DE METRICAS DE DESEMPEÑO
df_accuracy=pd.DataFrame()
df_recall=pd.DataFrame()

df_accuracy['knn']=knn_scores['test_accuracy'].tolist()
df_accuracy['logistic_regression']=lr_scores['test_accuracy'].tolist()
df_accuracy['svm']=svm_scores['test_accuracy'].tolist()
df_accuracy['mlp']=mlp_scores['test_accuracy'].tolist()
df_accuracy['random_forest']=rf_scores['test_accuracy'].tolist()
print(df_accuracy.mean(axis='index').sort_values())

df_recall['knn']=knn_scores['test_recall'].tolist()
df_recall['logistic_regression']=lr_scores['test_recall'].tolist()
df_recall['svm']=svm_scores['test_recall'].tolist()
df_recall['mlp']=mlp_scores['test_recall'].tolist()
df_recall['random_forest']=rf_scores['test_recall'].tolist()
print(df_recall.mean(axis='index').sort_values())

#Generar boxplot de accuracy
p = sns.boxplot(data = df_accuracy)
p.set_xlabel('Algorithm', fontsize= 14, fontweight='bold')
p.set_ylabel('Accuracy', fontsize= 14, fontweight='bold')
p.set_title('Accuracy Scores for Used Algorithms', fontsize= 16, fontweight='bold');
plt.show()

#Generar boxplot de recall
p = sns.boxplot(data = df_recall)
p.set_xlabel('Algorithm', fontsize= 14, fontweight='bold')
p.set_ylabel('Recall', fontsize= 14, fontweight='bold')
p.set_title('Recall Scores for Used Algorithms', fontsize= 16, fontweight='bold');
plt.show()

#Con base en las métricas de accuracy y recall, los mejores modelos son MLP y KNN
#La tarea pide la matriz de confusion de los mejores modelos, ya se habían hecho,
#entonces solo hay que arbir o pegar la matriz de KNN y MLP

#c. Obtener la matriz de confusión del mejor modelo obtenido en el inciso anterior. 
# Interpreta dicha matriz, en particular comenta sobre los falsos positivos 
# y falsos negativos obtenidos. ¿Cuál consideras que pudiera ser el error más importante, 
# con base al contexto del problema y desde el punto de vista de la institución bancaria?


#d. Incluye tus conclusiones de esta sección, 
# en particular menciona cuáles son los mejores 3 modelos obtenidos, 
# con base a toda la información obtenida.


#Parte III: Métodos para clases no balanceadas
#4. En esta parte deberás trabajar con el que consideres 
# fue el mejor modelo obtenido en la Parte II.
#Vamos a trabajar con RF y MLP

#a. Utiliza al menos 5 técnicas diferentes para clases no balanceadas 
# en combinación con el mejor modelo seleccionado para entrenar el modelo 
# y utilizando las métricas “accuracy” y “recall”. Debes utilizar validación cruzada.

#Estrategia 1: Ajuste de Metricas Weight
#Aplicada a KNN
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors,weights="distance")
knn.fit(X_train, y_train)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train, y_train)))
y_pred = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)
precision_knn = precision_score(y_test, y_pred)
recall_knn = recall_score(y_test, y_pred)
f1score_knn = f1_score(y_test, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier with weight adjustment', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
knn_scores_weight = cross_validate(knn, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN with weight adjustment', knn_scores_weight)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores_weight['test_accuracy'].min(), knn_scores_weight['test_accuracy'].mean(), knn_scores_weight['test_accuracy'].max()))

#Estrategia 2: Subsampling en la clase mayoritaria
#Aplicado a KNN
us = NearMiss(n_neighbors=3, version=2)
X_train_res, y_train_res = us.fit_resample(X_train, y_train)
X_test_res, y_test_res = us.fit_resample(X_test, y_test)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_res, y_train_res)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train_res, y_train_res)))
y_pred = knn.predict(X_test_res)
accuracy_knn = accuracy_score(y_test_res, y_pred)
precision_knn = precision_score(y_test_res, y_pred)
recall_knn = recall_score(y_test_res, y_pred)
f1score_knn = f1_score(y_test_res, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test_res, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test_res, y_test_res, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier with Subsampling', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
X_res, y_res = us.fit_resample(X, y)
knn_scores_subsampling = cross_validate(knn, X_res, y_res, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN with Subsampling', knn_scores_subsampling)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores_subsampling['test_accuracy'].min(), knn_scores_subsampling['test_accuracy'].mean(), knn_scores_subsampling['test_accuracy'].max()))

#Estrategia 3: Oversampling de la clase minoritaria
#Aplicado a KNN
os =  RandomOverSampler()
X_train_res, y_train_res = os.fit_resample(X_train, y_train)
X_test_res, y_test_res = os.fit_resample(X_test, y_test)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_res, y_train_res)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train_res, y_train_res)))
y_pred = knn.predict(X_test_res)
accuracy_knn = accuracy_score(y_test_res, y_pred)
precision_knn = precision_score(y_test_res, y_pred)
recall_knn = recall_score(y_test_res, y_pred)
f1score_knn = f1_score(y_test_res, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test_res, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test_res, y_test_res, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier with Oversampling', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
X_res, y_res = os.fit_resample(X, y)
knn_scores_oversampling = cross_validate(knn, X_res, y_res, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN with Oversampling', knn_scores_oversampling)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores_oversampling['test_accuracy'].min(), knn_scores_oversampling['test_accuracy'].mean(), knn_scores_oversampling['test_accuracy'].max()))

#Estrategia 4: Combinamos resampling con Smote-Tomek
#Aplicado a KNN
os_us = SMOTETomek()
X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)
X_test_res, y_test_res = os_us.fit_resample(X_test, y_test)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_res, y_train_res)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train_res, y_train_res)))
y_pred = knn.predict(X_test_res)
accuracy_knn = accuracy_score(y_test_res, y_pred)
precision_knn = precision_score(y_test_res, y_pred)
recall_knn = recall_score(y_test_res, y_pred)
f1score_knn = f1_score(y_test_res, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test_res, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test_res, y_test_res, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier with SMOTE', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
X_res, y_res = os_us.fit_resample(X, y)
knn_scores_smote = cross_validate(knn, X_res, y_res, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN with SMOTE', knn_scores_smote)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores_smote['test_accuracy'].min(), knn_scores_smote['test_accuracy'].mean(), knn_scores_smote['test_accuracy'].max()))

#Estrategia 5: Adaptive Synthetic Sampling (ADASYN)
#Aplicado a KNN
oversample=ADASYN()
X_train_res, y_train_res = oversample.fit_resample(X_train, y_train)
X_test_res, y_test_res = oversample.fit_resample(X_test, y_test)
print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_res, y_train_res)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train_res, y_train_res)))
y_pred = knn.predict(X_test_res)
accuracy_knn = accuracy_score(y_test_res, y_pred)
precision_knn = precision_score(y_test_res, y_pred)
recall_knn = recall_score(y_test_res, y_pred)
f1score_knn = f1_score(y_test_res, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test_res, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test_res, y_test_res, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier with ADASYN', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the general performance of the model
X_res, y_res = os_us.fit_resample(X, y)
knn_scores_adasyn = cross_validate(knn, X_res, y_res, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN with ADSYN', knn_scores_adasyn)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores_adasyn['test_accuracy'].min(), knn_scores_adasyn['test_accuracy'].mean(), knn_scores_adasyn['test_accuracy'].max()))

#b. Obtener un diagrama de cajas conjunto de todos los modelos con los 
# resultados de la métrica “accuracy” y otro diagrama de cajas conjunto 
# los resultados de la métrica “recall”.

#ARMAR DATASET CON LOS SCORES PARA HACER BOXPLOT DE METRICAS DE DESEMPEÑO DE KNN
df_accuracy_knn=pd.DataFrame()
df_recall_knn=pd.DataFrame()

df_accuracy_knn['knn_simple']=knn_scores['test_accuracy'].tolist()
df_accuracy_knn['knn_weight']=knn_scores_weight['test_accuracy'].tolist()
df_accuracy_knn['knn_subsampling']=knn_scores_subsampling['test_accuracy'].tolist()
df_accuracy_knn['knn_oversampling']=knn_scores_oversampling['test_accuracy'].tolist()
df_accuracy_knn['knn_smote']=knn_scores_smote['test_accuracy'].tolist()
df_accuracy_knn['knn_adasyn']=knn_scores_adasyn['test_accuracy'].tolist()
print(df_accuracy_knn.mean(axis='index').sort_values())

df_recall_knn['knn_simple']=knn_scores['test_recall'].tolist()
df_recall_knn['knn_weight']=knn_scores_weight['test_recall'].tolist()
df_recall_knn['knn_subsampling']=knn_scores_subsampling['test_recall'].tolist()
df_recall_knn['knn_oversampling']=knn_scores_oversampling['test_recall'].tolist()
df_recall_knn['knn_smote']=knn_scores_smote['test_recall'].tolist()
df_recall_knn['knn_adasyn']=knn_scores_adasyn['test_recall'].tolist()
print(df_recall_knn.mean(axis='index').sort_values())

#Generar boxplot de accuracy
p = sns.boxplot(data = df_accuracy_knn)
p.set_xlabel('Strategy', fontsize= 14, fontweight='bold')
p.set_ylabel('Accuracy', fontsize= 14, fontweight='bold')
p.set_title('Accuracy Scores for KNN with inbalanced class strategies', fontsize= 16, fontweight='bold');
plt.show()

#Generar boxplot de recall
p = sns.boxplot(data = df_recall_knn)
p.set_xlabel('Stratehy', fontsize= 14, fontweight='bold')
p.set_ylabel('Recall', fontsize= 14, fontweight='bold')
p.set_title('Recall Scores for KNN with inbalanced class strategies', fontsize= 16, fontweight='bold');
plt.show()

#c. Incluye tus conclusiones de los resultados obtenidos.


#Parte IV: Mejor modelo
#El mejor modelo fue el knn simple, ahora vamos a buscar los mejores 
# parametros para ejecutarlo

n_neighbors = 12
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print("\n METRICAS EVALUACIÓN KNN \n")
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train, y_train)))
y_pred = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)
precision_knn = precision_score(y_test, y_pred)
recall_knn = recall_score(y_test, y_pred)
f1score_knn = f1_score(y_test, y_pred)
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(accuracy_knn))
print('Precision of K-NN classifier on test set: {:.4f}'
     .format(precision_knn))
print('Recall of K-NN classifier on test set: {:.4f}'
     .format(recall_knn))
print('F1_Score of K-NN classifier on test set: {:.4f}'
     .format(f1score_knn))
result_knn = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix for KNN classifier:")
print(result_knn)
color = 'black'
matrix = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix for KNN classifier using K=12', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#Using Cross Validation to Get the Best Value of k
k_values = [i for i in range (1,31)]
scores = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_validate(knn, X, y, cv=cv_iterations, scoring=metrics)
    scores.append(np.mean(score['test_accuracy']))
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()
#EL MEJOR VALOR PARA K ES 12, PORQUE QES EL VALOR MAS PEQUEÑO QUE TIENE MEJOR ACCURACY
#EL MEJOR DESEMPEÑO DE KNN PARA ESTE DATASET Y ALGORITMO ES CON K=12 
#CON ESE VALOR DE K SE ALCANZA UN ACCURACY DE 0.77
#Using Cross Validation to Get the general performance of the model
#Vamos a hacer el crossvalidation ya con k=12
knn_scores = cross_validate(knn, X, y, cv=cv_iterations, scoring=metrics)
print('\n Cross-Validation Accuracy Scores for KNN', knn_scores)
#knn_scores = pd.Series(knn_scores)
print('\n Min score for KNN: {:.4f} \n Mean score for KNN: {:.4f} \n Max score for KNN: {:.4f}' .format(knn_scores['test_accuracy'].min(), knn_scores['test_accuracy'].mean(), knn_scores['test_accuracy'].max()))
