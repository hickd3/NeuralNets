'''softmax_layer.py
Constructs, trains, tests single layer neural network with softmax activation function.
Dean Hickman
CS 343: Neural Networks
Spring 2025
Project 2: Multilayer Perceptrons
'''
import numpy as np


class SoftmaxLayer:
    '''SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        accuracy = (np.sum(y_pred == y) / y.shape[0])
        return accuracy 

    def net_in(self, features):
        '''Computes the net input (net weighted sum)

        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        net_input= (np.dot(features, self.wts))+ self.b
        return net_input


    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        y_one_hot = np.zeros((y.shape[0], num_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def fit(self, features, y, n_epochs=100, lr=0.0001, mini_batch_sz=256, reg=0, r_seed=None, verbose=2):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        r_seed: None or int. Random seed for weight and bias initialization.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and
            training loss every 100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.

        NOTE:
        Recall: training epoch is not the same thing as training iteration with mini-batch.
        If we have mini_batch_sz = 100 and N = 1000, then we have 10 iterations per epoch. Epoch
        still means entire pass through the training data "on average". Print this information out
        if verbose > 0.

        TODO:
        -----------
        1) Initialize the wts/bias to small Gaussian numbers:
            mean 0, std 0.01, Wts shape=(num_feat M, num_classes C), b shape=(num_classes C,)
        2) Implement mini-batch support: On every iter draw from our input samples (with replacement)
        a batch of samples equal in size to `mini_batch_sz`. Also keep track of the associated labels.
        THEY MUST MATCH UP!!
            - Keep in mind that mini-batch wt updates are different than epochs. There is a parameter
              for E (num epochs), not number of iterations.
            - Handle this edge case: we do SGD and mini_batch_sz = 1. Add a singleton dimension
              so that the "N"/sample_size dimension is still defined.
        4) Our labels are int coded (0,1,2,3...) but this representation doesnt work well for piping
        signals to the C output neurons (C = num classes). Transform the mini-batch labels to one-hot
        coding from int coding (see function above to write this code).
        5) Compute the "net in".
        6) Compute the activation values for the output neurons (you can defer the actual function
        implementation of this for later).
        7) Compute the cross-entropy loss (again, you can defer the details for now)
        8) Do backprop:
            a) Compute the error gradient for the mini-batch sample,
            b) update weights using gradient descent.

        HINTS:
        -----------
        2) Work in indices, not data elements.
        '''
        if r_seed is not None:
            np.random.seed(r_seed)

    
        self.wts, self.b = np.random.normal(0, 0.01, (features.shape[1], self.num_output_units)), np.random.normal(0, 0.01, self.num_output_units)

        loss_history = []
        N = features.shape[0]
        num_batches = int(np.ceil(N / mini_batch_sz))
        total_iter = n_epochs * num_batches

        if verbose > 0:
            print(f"Starting to train network...There will be {n_epochs} epochs and {total_iter} iterations total, {num_batches} iter/epoch.")

        iter_count = 0
        for epoch in range(n_epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            shuffled_X = features[indices]
            shuffled_y = y[indices]

            for i in range(num_batches):
                start_idx = i * mini_batch_sz
                end_idx = min((i + 1) * mini_batch_sz,N)

                batch_features = shuffled_X[start_idx:end_idx]
                batch_labels_int = shuffled_y[start_idx:end_idx]
                
                if batch_features.ndim == 1:
                    batch_features = batch_features[np.newaxis, :]
                    batch_labels_int = np.array([batch_labels_int])

                batch_labels_1hot = self.one_hot(batch_labels_int, self.num_output_units)

                net_in = self.net_in(batch_features)
                net_act = self.activation(net_in)

                loss = self.loss(net_act, batch_labels_int, reg)
                loss_history.append(loss)

                grad_wts, grad_b = self.gradient(batch_features, net_act, batch_labels_1hot, reg)
                self.wts -= lr * grad_wts
                self.b -= lr * grad_b

                if verbose > 0 and iter_count % 100 == 0:
                    print(f"  Completed iter {iter_count}/{total_iter}. Training loss: {loss:.2f}.")

            iter_count += 1

        if verbose > 0:
            print("Finished training!")
        
        return loss_history


    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        net_in = self.net_in(features)
        net_act = self.activation(net_in)
        y_pred= np.argmax(net_act, axis=1)
        return y_pred

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        net_in = net_in - np.max(net_in, axis=1, keepdims=True)
        expnet_in = np.exp(net_in)
        f_z = expnet_in / np.sum(expnet_in, axis=1, keepdims=True)
        return f_z
        

    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to select only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        '''
        N = y.shape[0] 
        pos_class_proba = -np.log(net_act[np.arange(N), y])
        cross_entropy_loss = 0.5 * reg * np.sum(self.wts * self.wts) + np.sum(pos_class_proba) / N
        return cross_entropy_loss 

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''
        errors = net_act - y
        N = features.shape[0]
        grad_bias = np.sum(errors, axis =0)  / N
        grad_wts = (np.dot(features.T, errors)  / N)+reg * self.wts
        return grad_wts, grad_bias

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.584664799299611, max=1.411396670099296\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.0665134672262976, max=0.1439281981621258\n')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are:\n{net_in[0, :5]}\nand should be')
        print('[0.798 1.095 0.969 0.9   0.958]')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are:\n{net_act[0, :5]}\nand should be')
        print('[0.078 0.105 0.092 0.086 0.091]')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}.')
        print('You should know what the sum should be :)')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
