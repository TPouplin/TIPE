import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def B(x): # Condition initiale
    cond = tf.cast(x <= 0.2, tf.float32)
    return (0.65 * cond) + ((1-cond) * 0.9)

def df(x): # Dérivé du flux 
    return 1-tf.exp(1-1/x) + (-1/(x))*tf.exp(1 - 1/x)

def Init(x): # permet de séparer les points qui sont contraints par les conditions initiales
    return (tf.cond(x <= 0, lambda: [1.], lambda:[0.]))

# Définition de l'espace : 
nx = 20
ny = 20
x_space = (np.linspace(0., 1., nx))
y_space = (np.linspace(0., 1., ny))

# Neurones par couche ( on a 2 couches) : 
n_nodes_hl1 = 20
n_nodes_hl2 = 20
x = tf.compat.v1.placeholder('float', [None, 2])

def neural_network_model(data, grad_coef=10): # Initialisation du réseau    
    # Mise en place du réseau :        
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, 1])),
                    'biases': tf.Variable(tf.random_normal([1]))}

    l1 = tf.sigmoid(tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases']))
    l2 = tf.sigmoid(tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases']))
    output = tf.add(tf.matmul(l2, output_layer['weights']),output_layer['biases'])
    trial = tf.sigmoid(output)  

    # Implémentation du coût :
    d = tf.gradients(trial, data)[0]
    cost = tf.add(tf.multiply(d[:, 0], f(trial)), d[:, 1])
    cost = tf.square(cost)
    cost = tf.reduce_mean(cost)
    
    # Contrainte des conditions initiales :
    additional_cost =[tf.reduce_sum(tf.multiply(tf.transpose([tf.map_fn(aux,data[:,1])]),(tf.square(trial-B(tf.transpose([data[:,0]]))))))]
    
    # Coût final :
    cost += grad_coef * additional_cost    
    return [trial,cost]

def train_neural_network(x): # Apprentissage 

    net_out, cst= neural_network_model(x)  
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cst) # Choix de l'algorithme d'optimisation
    hm_epochs = 1001 #Nombre de période d'apprentissage
    
    fd = {x: [[xi, yi] for xi in x_space for yi in y_space]}
    
    with tf.Session() as sess: # Boucle d'apprentissage
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            _, epoch_loss = sess.run([optimizer, cst],feed_dict=fd)
            print(sum(epoch_loss))