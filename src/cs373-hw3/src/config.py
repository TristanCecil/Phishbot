import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f_dim', type=int, default=10000, help='Dimension of the input feature vector to the classifier')
parser.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary')
parser.add_argument('--num_iter', type=int, default=50, help='Number of iterations to be run for training')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate of perceptron')
parser.add_argument('--bin_feats', type=bool, default=False, help='Use binary word features')
parser.add_argument('--saved_model_legit', default=None, help='destination of saved legitimate model')
parser.add_argument('--saved_model_phish', default=None, help='destination of saved phishing model')
#Add any arguments you would like to pass to the classifier (all hyperparameters go here!)
args = parser.parse_args()
