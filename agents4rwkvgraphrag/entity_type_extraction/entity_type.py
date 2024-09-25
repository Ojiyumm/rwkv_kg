from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch

# download models: https://huggingface.co/BlinkDL
model = RWKV(model='/home/rwkv/Peter/model/base/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda fp16')
print(model.args)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# use pipeline = PIPELINE(model, "rwkv_vocab_v20230424") for rwkv "world" models
states_file = '/home/rwkv/Peter/rwkv_graphrag/agents/entity_type_extraction/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
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
instruction ='根据input中的领域和任务，协助用户识别input文本中存在的实体类型。 实体类型必须与用户任务相关。 避免使用诸如“其他”或“未知”的通用实体类型。 非常重要的是：不要生成冗余或重叠的实体类型。用JSON格式输出。'
input_text = '{"领域": "文学与神话", "专家": "文学史学者/神话学家", "任务": ["分析《石头记》的历史背景和影响", "研究《红楼梦》与《金陵十二钗》之间的关系", "探讨东鲁孔梅溪对《石头记》的改编过程", "解析吴玉峰在《红楼梦》中的角色和贡献", "评估曹雪芹在《悼红轩中披阅十五间》中的写作技巧"]}'
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)

def my_print(s):
    print(s, end='', flush=True)



args = PIPELINE_ARGS(temperature = 1, top_p = 0.2, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.5,
                     alpha_presence = 0.5,
                     alpha_decay = 0.998, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [0,1], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print,state=states_value)
print('\n')    