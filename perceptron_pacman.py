# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # Coger lo que se necesita
                etiquetasGeneral = self.legalLabels  # Coger las etiquetas (en general)
                etiquetaTData = trainingLabels[i]  # Coger las etiquetas de cada instancia
                instancia = trainingData[i]  # Coger el valor de cada instancia del trainingData

                score = util.Counter()
                for etiqueta in etiquetasGeneral:
                    score[etiqueta] = self.weights[etiqueta] * instancia  # Multiplica el peso actual por el valor (y'')

                # Toca sacar el y' (el mayor peso escalar)
                # scoreMax = np.argmax(score) # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
                scoreMax = score.argMax()  # No me habia dado cuenta de que tiene un metodo propio para argMax
                if (scoreMax != etiquetaTData):  # Si la predicha es difente a la real cambiar los pesos
                    self.weights[scoreMax] -= instancia  # A la que ha predicho hay que quitarle peso
                    self.weights[etiquetaTData] += instancia  # Hay que sumarle peso a la real
