import token_to_vocab
from collections import defaultdict
class frequencyCounter():
    def __init__(self):
        self.totalcounts = defaultdict(int)
        self.tokens = []
    """
    def buildVocab(self, filepath):
        file = open(filepath, "r", encoding="utf8", errors="namespace").read()
        self.tokens = file.split("\n")
    """

    def getFeatureVector(self, ex):
        counts = defaultdict(int)
        #global tokens
        for t in token_to_vocab.tokens:
            counts[t] += self.frequencyOfToken(ex, t)
            #print(counts)
            self.totalcounts[t] += counts[t]
        return counts.values()



    def frequencyOfToken(self, ex, token):
        return ex.count(token)
