import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.size  = 0
        self.max_vocab = 0
        
    def create_vocab(self,sent):
        mx = 0 
        sentences = [nltk.tokenize.word_tokenize(text) for text in sent]
        vocab = []
        for sents in sentences:
            self.size = max(self.size,len(sents))
            for words in sents:
                if(words not in vocab):
                    vocab.append(words)
        for i in enumerate(vocab):
            self.vocab[i[1]] = i[0]
            self.max_vocab = i[0]+1
    
    def tokenize(self,texts):
        sentences = torch.zeros(len(texts),self.size,dtype=torch.int16)
        i = 0 
        for sents in texts:
            words = nltk.tokenize.word_tokenize(sents)
            j = 0
            for word in words:
                try:
                    sentences[i][j] = self.vocab[word] 
                except:
                    sentences[i][j] = -1
                j+=1
                if(j==self.size):
                    break
            i+=1
        return sentences
