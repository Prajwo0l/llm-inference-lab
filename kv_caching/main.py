import argparse
import torch 
from trainer import train 
from inference import generate_text , benchmark



def main():
    parser = argparse.ArgumentParser(description = "GPT from scartch using kv cache")
    parser.add_argument("--iters",type=int,default=500,help="Training iterations")
    parser.add_argument("--tokens",type=int,default=200,help="Tokens to generate")
    parser.add_argument("--prompt", type=str,default="To be", help="Generation prompt")
    args=parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #train
    model , dataset = train(max_iters=args.iters,device=device)
    # generate text (kv cache)
    print("\n-- Generated text(KV CACHE)")
    text = generate_text(
        model,dataset,
        prompt=args.prompt,
        temperature=0.8,
        top_k=40,
        use_kv_cache=True,
    )
    print(text)

    # benchmark kv cache vs naive
    benchmark(model,dataset,prompt=args.prompt ,new_tokens=100,runs=3)

if __name__=="__main__":
    main()
    