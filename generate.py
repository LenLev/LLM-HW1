import torch 
from transformers import AutoTokenizer, Qwen3ForCausalLM 
 
CHECKPOINT = "./output_dir/gpt2-1b-russian/checkpoint-1429" 
 
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT) 
 
model = Qwen3ForCausalLM.from_pretrained( 
    CHECKPOINT, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2" 
) 
 
model = model.cuda() 
model.eval() 
 
prompt = "Эмпатия это" 
 
inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 
 
with torch.no_grad(): 
    outputs = model.generate( 
        **inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.8, 
        top_p=0.9 
    ) 
 
text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
 
print("\nPROMPT") 
print(prompt) 
print("\nGENERATED TEXT") 
print(text)
