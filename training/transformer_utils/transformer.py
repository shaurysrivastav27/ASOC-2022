import torch


class Transformers(torch.nn.Module):
    
    def __init__(self,layers,max_vocab,embed_size,classes,num_max=500):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_vocab+1,embed_size)
        self.position_embedding = torch.nn.Embedding(num_max+1,embed_size)
        self.attention,self.sequential = torch.nn.ModuleList(),torch.nn.ModuleList()
        self.norm2 , self.norm1 = torch.nn.ModuleList(),torch.nn.ModuleList()
        
        for _ in range(layers):
            self.attention.append(torch.nn.MultiheadAttention(embed_size,5))
            layer1 = torch.nn.Linear(embed_size,10)
            layer2 = torch.nn.Linear(10,embed_size)
            torch.nn.init.xavier_normal_(layer1.weight)
            torch.nn.init.xavier_normal_(layer2.weight)
            self.sequential.append(torch.nn.Sequential(layer1,torch.nn.ReLU(inplace=True),layer2))
            self.norm1.append(torch.nn.LayerNorm(embed_size))
            self.norm2.append(torch.nn.LayerNorm(embed_size))
        
    def forward(self,x):
        positions = torch.arange(0,500)
        h = self.embedding(x)
        h = h + self.position_embedding(positions).expand_as(h)
        
        for norm1,attention,linear,norm2 in zip(self.norm1,self.attention,self.sequential,self.norm2):
            h = norm1(h)
            x,_ = attention(h,h,h,need_weights=False)
            h = x+h
            h = norm2(h)
            x = linear(h)
            h = h+x
        return h
