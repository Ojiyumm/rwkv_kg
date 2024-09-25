from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch

# download models: https://huggingface.co/BlinkDL
model = RWKV(model='/home/rwkv/Peter/model/base/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda fp16')
print(model.args)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# use pipeline = PIPELINE(model, "rwkv_vocab_v20230424") for rwkv "world" models
states_file = '/home/rwkv/Peter/rwkv_graphrag/agents/entity_summary/entity_summary.pth'
states = torch.load(states_file)
states_value = []
device = 'cuda'
n_head = model.args.n_head
head_size = model.args.n_embd//model.args.n_head
for i in range(model.args.n_layer):
    key = f'blocks.{i}.att.time_state'
    value = states[key]
    prev_x = torch.zeros(model.args.n_embd,device=device,dtype=torch.float16)
    prev_states = value.clone().detach().to(device=device,dtype=torch.float16).transpose(1,2)
    prev_ffn = torch.zeros(model.args.n_embd,device=device,dtype=torch.float16)
    states_value.append(prev_x)
    states_value.append(prev_states)
    states_value.append(prev_ffn)

cat_char = '🐱'
bot_char = '🤖'
instruction ='请阅读iput中的entities和descritions，围绕entity和descrition写一个简单的200字介绍，介绍需要包括所的entity和他们之间的关系.最终内容不能超过200字'
input_text = '"entities": ["汉语", "语义偏移", "构式语法", "评价性语境", "词汇意义"], "descriptions": ["汉语是中国的主要语言，具有丰富的语义结构和复杂的语法体系。", "语义偏移是指在特定语境下，词语的意义发生的变化或偏离。", "构式语法研究的是句子结构的模式及其功能，是语言学的一个分支。", "评价性语境指的是包含情感色彩或评价性质的语言环境，影响着语言表达的意义。", "词汇意义指的是单词在特定语境下的具体含义，可以因语境而变化。"]'
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)

def my_print(s):
    print(s, end='', flush=True)



args = PIPELINE_ARGS(temperature = 1.3, top_p = 0.5, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.7,
                     alpha_presence = 0.5,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [bot_char], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print,state=states_value)
print('\n')    