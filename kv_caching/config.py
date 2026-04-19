class GPTConfig:
    #Vocabulary and context
    vocab_size:int =65
    block_size : int=128

    #Model size
    n_embed:int=192  #embedding dimension
    n_head:int=6    #number of attentuion heads
    n_layer:int=4 # number of stacked transfoermer blocks


    #Regularizatioon
    dropout:float=0.1
    bias:bool =True #using bias in linear and layernorm layers

