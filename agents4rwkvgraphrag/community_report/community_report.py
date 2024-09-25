from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch

# download models: https://huggingface.co/BlinkDL
model = RWKV(model='/home/rwkv/Peter/model/base/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda fp16')
print(model.args)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# use pipeline = PIPELINE(model, "rwkv_vocab_v20230424") for rwkv "world" models
states_file = '/home/rwkv/Peter/rwkv_graphrag/agents/community_report/community_report_2048.pth'
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

import json
instruction ='"给你一段社区中命名体和他们之间关系的描述，作文文学家请写出一种关于这个社区的总结报告，报告需要包括title，summary，rating，rating explaination，findings和reference。'
#input_text="{\"entities\": [{\"entity\": \"RFID系统\", \"description\": \"RFID模型是车联网仿真的基础和核心模型之一.\", \"id\": 1}, {\"entity\": \"读写器\", \"description\": \"建立了RFID系统的读写器模型，并给出了读写器Agent的结构及实现.\", \"id\": 2}, {\"entity\": \"电子标签\", \"description\": \"建立了RFID系统的电子标签模型，并给出了电子标签Agent的结构及实现.\", \"id\": 3}, {\"entity\": \"通信协议\", \"description\": \"根据自动机理论，建立了读写器、电子标签和主机之间通信时异构通信协议数据帧格式的有限状态机模型.\", \"id\": 4}, {\"entity\": \"有限状态机模型\", \"description\": \"解决了异构数据帧统一处理问题，通过该模型实现了数据帧的统一处理.\", \"id\": 5}, {\"entity\": \"车联网仿真\", \"description\": \"作为模型应用和验证实例，基于VC++2010平台，开发了RFID交通数据采集读写器优化布设仿真软件.\", \"id\": 6}, {\"entity\": \"数据采集\", \"description\": \"开发的仿真软件用于RFID交通数据采集，优化布设读写器.\", \"id\": 7}, {\"entity\": \"优化布设\", \"description\": \"基于仿真软件，对RFID系统的读写器进行优化布设.\", \"id\": 8}, {\"entity\": \"软件开发\", \"description\": \"基于VC++2010平台，开发了RFID交通数据采集读写器优化布设仿真软件.\", \"id\": 9}, {\"entity\": \"VC++2010平台\", \"description\": \"在VC++2010平台上进行软件开发，以实现RFID系统的相关功能.\", \"id\": 10}, {\"entity\": \"物理设备\", \"description\": \"RFID模型较好地逼近了RFID物理设备，适用于车联网场景.\", \"id\": 11}, {\"entity\": \"仿真软件\", \"description\": \"开发的基于VC++2010平台的仿真软件，用于验证RFID模型在车联网场景下的性能和适用性.\", \"id\": 12}, {\"entity\": \"开发基础\", \"description\": \"整个开发过程基于一定的技术基础和理论知识，确保模型和软件的有效性和实用性.\", \"id\": 13}], \"relationships\": [{\"source\": \"RFID系统\", \"target\": \"读写器\", \"relationship\": \"读写器是RFID系统的关键组成部分，用于与电子标签进行数据交换.\", \"id\": 14}, {\"source\": \"RFID系统\", \"target\": \"通信协议\", \"relationship\": \"通信协议定义了读写器和电子标签之间的数据传输规则，是RFID系统正常运行的基础.\", \"id\": 15}, {\"source\": \"读写器\", \"target\": \"电子标签\", \"relationship\": \"读写器与电子标签进行双向通信，实现数据的读取和写入操作.\", \"id\": 16}, {\"source\": \"车联网仿真\", \"target\": \"仿真软件\", \"relationship\": \"仿真软件用于模拟和验证RFID系统在车联网环境中的性能和效果.\", \"id\": 17}, {\"source\": \"优化布设\", \"target\": \"读写器\", \"relationship\": \"优化布设策略旨在提高RFID系统的效率和覆盖范围，通过合理布置读写器实现这一目标.\", \"id\": 18}, {\"source\": \"软件开发\", \"target\": \"VC++2010平台\", \"relationship\": \"VC++2010平台提供了一个高效、稳定的开发环境，支持RFID系统相关软件的开发.\", \"id\": 19}]}"
input_text = "{'entities': [{'entity': '空空道人', 'description': '一位从大荒山无稽崖青埂峰下经过的访道求仙者，他在看到《石头记》编述历程时，将其改名为《情僧录》。', 'id': 1}, {'entity': '石头记', 'description': '《石头记》是一部由曹雪芹编写的小说，讲述了贾宝玉、林黛玉等人的故事。', 'id': 2}, {'entity': '情僧录', 'description': '《情僧录》是空空道人改编的《石头记》的新名字，旨在突出其中的情感元素。', 'id': 3}],'relationships': [{'source': '空空道人', 'target': '《石头记》', 'relationship': '空空道人对《石头记》进行了改编和重新命名。'}, {'source': '空空道人', 'target': '情僧录', 'relationship': '空空道人将《石头记》改名为《情僧录》。'}, {'source': '空空道人', 'target': '红楼梦', 'relationship': '空空道人对《红楼梦》进行了深入的研究和评价。'} {'source': '甄士隐', 'target': '贾雨村', 'relationship': '甄士隐与贾雨村是朋友关系。'}, {'source': '甄士隐', 'target': '封肃家', 'relationship': '甄士隐在封肃家中居住并接济了贾雨村。'}, {'source': '甄士隐', 'target': '胡州人氏贾化（号雨村）', 'relationship': '甄士隐与贾雨村有着相似的主题。'}]}"
ctx = f'Instruction: {instruction}\nInput: {input_text}\n\nResponse:'
print(ctx)


def my_print(s):
    print(s, end='', flush=True)



args = PIPELINE_ARGS(temperature = 1.2, top_p = 0.2, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [261], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

# stop_tok='\n\n'
# print(pipeline.encode(stop_tok))
pipeline.generate(ctx, token_count=2000, args=args, callback=my_print,state=states_value)

print('\n')    