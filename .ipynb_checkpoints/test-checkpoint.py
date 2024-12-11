# test
from model_merging.data import load_glue_dataset
from transformers import AutoTokenizer

def main():
    model_str = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    ds = load_glue_dataset(
        task="sst-2",        #Can be replaced by other tasks
        split="validation",
        tokenizer=tokenizer,
        max_length=128,
    )
    for batch in ds.take(1):
        print(batch)

if __name__ == "__main__":
    main()
