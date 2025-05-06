import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

word2idx = {}
label2idx = {}
class CustomDataset(Dataset): 
    def __init__(self,filepath):
        self.sentences , self.ner_tags = self.load_data(filepath)
        
    def load_data(self,filepath):
        sentences, ner_tags = [],[]
        sentence, ner_tag = [],[]
        with open(filepath,"r",encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    word,_,ner = line.split("\t")
                    sentence.append(word)
                    ner_tag.append(ner)
                else:
                    if sentence:
                        sentences.append(sentence)
                        ner_tags.append(ner_tag)
            return sentences,ner_tags
                
filepath = "datasets/train_v5.conll"

train_data = CustomDataset(filepath)
sentences, ner_tags = train_data.sentences, train_data.ner_tags

# Print first 3 samples
for i in range(3):
    print("Sentence:", sentences[i])
    print("NER Tags:", ner_tags[i])
    print("-" * 10)
