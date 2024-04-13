import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        score_float = nn.as_scalar(score)
        if score_float >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            flag = True
            for x, y in dataset.iterate_once(batch_size):
                y_hat = self.get_prediction(x)
                if y_hat * y.data < 0:
                    weights = self.get_weights()
                    # 感知机仅通过数据x和标签来更新权重
                    weights.update(x,nn.as_scalar(y))
                    flag = False
                # print(self.get_weights().data)
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        # First Layers
        self.m1 = nn.Parameter(1,64)
        self.b1 = nn.Parameter(1,64)
        
        # Second Layers
        self.m2 = nn.Parameter(64,32)
        self.b2 = nn.Parameter(1,32)
        
        # Output Layers
        self.m3 = nn.Parameter(32,1)
        self.b3 = nn.Parameter(1,1)
        self.param = [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x,self.m1),self.b1))
        
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer,self.m2),self.b2))
        
        output_layer = nn.AddBias(nn.Linear(second_layer,self.m3),self.b3)

        return output_layer
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SquareLoss(predicted_y,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        loss = 1e9
        while loss >= 0.01:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss,self.param)
                for i in range(len(grad)):
                    self.param[i].update(grad[i],-self.learning_rate)
                loss = nn.as_scalar(loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # 第一层参数
        self.w1 = nn.Parameter(784,256)
        self.b1 = nn.Parameter(1,256)
        
        # 第二层参数
        self.w2 = nn.Parameter(256,128)
        self.b2 = nn.Parameter(1,128)
        
        # 第三层参数
        self.w3 = nn.Parameter(128,10)
        self.b3 = nn.Parameter(1,10)
        
        self.param = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        self.learning_rate = 0.5
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # First_layer
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x,self.w1),self.b1))
        
        # Second_layer
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer,self.w2),self.b2))
        
        # Third_layer
        third_layer = nn.AddBias(nn.Linear(second_layer,self.w3),self.b3)
        
        return third_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_predict = self.run(x)
        loss = nn.SoftmaxLoss(y_predict,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        loss = 1e9
        while loss >= 0.015:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss,self.param)
                for i in range(len(grad)):
                    self.param[i].update(grad[i],-self.learning_rate)
                loss = nn.as_scalar(loss)
            

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.initial_wx = nn.Parameter(self.num_chars,256)
        self.initial_b = nn.Parameter(1,256)
        
        self.wx = nn.Parameter(self.num_chars,256)
        self.xb = nn.Parameter(1,256)
        self.wh = nn.Parameter(256,256)
        
        self.wy = nn.Parameter(256,len(self.languages))
        self.yb = nn.Parameter(1,len(self.languages))
        
        self.param = [self.initial_wx, self.initial_b, self.wx, self.xb, self.wy, self.yb]
        self.learning_rate = 0.01
        
        

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h_t = nn.AddBias(nn.Linear(xs[0],self.initial_wx),self.initial_b)
        for char in xs[1:]:
            h_t = nn.AddBias(nn.Add(nn.Linear(char,self.wx),nn.Linear(h_t,self.wh)),self.xb)
            h_t = nn.ReLU(h_t)
        output = nn.AddBias(nn.Linear(h_t,self.wy),self.yb)
        
        return output
        
            
            
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_predict = self.run(xs)
        loss = nn.SoftmaxLoss(y_predict,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        accuracy = 0
        while accuracy < 0.85:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss,self.param)
                for i in range(len(grad)):
                    self.param[i].update(grad[i],-self.learning_rate)
                loss = nn.as_scalar(loss)
            accuracy = dataset.get_validation_accuracy()