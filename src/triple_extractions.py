import json
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import argparse

def generate_triples(input_text, model_path, vocab_path, states_file, token_count=1500):
    # Load the model
    model = RWKV(model=model_path, strategy='cuda fp16')
    print(model.args)

    # Initialize the pipeline
    pipeline = PIPELINE(model, vocab_path)

    # Load previous states
    states = torch.load(states_file)
    states_value = []
    device = 'cuda'

    for i in range(model.args.n_layer):
        key = f'blocks.{i}.att.time_state'
        value = states[key]
        prev_x = torch.zeros(model.args.n_embd, device=device, dtype=torch.float16)
        prev_states = value.clone().detach().to(device=device, dtype=torch.float16).transpose(1, 2)
        prev_ffn = torch.zeros(model.args.n_embd, device=device, dtype=torch.float16)
        states_value.append(prev_x)
        states_value.append(prev_states)
        states_value.append(prev_ffn)

    # Create the context for the model
    instruction = '根据input中的input和entity_types，帮助用户找到文本中的entity_types的实体,标明实体类型并且简单描述，。然后给找到实体之间的关系，并且描述这段关系以及对关系强度打分。 避免使用诸如“其他”或“未知”的通用实体类型。 非常重要的是：不要生成冗余或重叠的实体类型和关系。用JSON格式输出。'
    ctx = f'Instruction: {instruction}\nInput: {input_text}\n\nResponse:'

    # Set up the generation arguments
    args = PIPELINE_ARGS(
        temperature=0.8,
        top_p=0.1,
        top_k=0,
        alpha_frequency=0.40,
        alpha_presence=0.31,
        alpha_decay=0.996,
        token_ban=[0],
        token_stop=[0, 1, 261],
        chunk_len=256
    )

    def my_print(s):
        print(s, end='', flush=True)

    # Generate the response
    triples = pipeline.generate(ctx, token_count=token_count, args=args, callback=my_print, state=states_value)
    return triples

def process_input(input_file, output_file, model_path, vocab_path, states_file, token_count=1500):
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            input_data = json.loads(line)
            torch.cuda.empty_cache()
            triples = generate_triples(input_data, model_path, vocab_path, states_file, token_count)
            triples = triples.replace("'", '"')
            formatted = {"triples": triples}

            with open(output_file, 'a', encoding='utf-8') as out_f:
                json.dump(formatted, out_f, ensure_ascii=False)
                out_f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate triples from input text.')
    parser.add_argument('--input_file', type=str, required=False, default='/home/rwkv/Peter/rwkv_graphrag/data/triple_input.jsonl')
    parser.add_argument('--output_file', type=str, required=False, default='/home/rwkv/Peter/rwkv_graphrag/data/triples.jsonl')
    parser.add_argument('--model_path', type=str, required=False, default='/home/rwkv/Peter/models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth')
    parser.add_argument('--vocab_path', type=str, required=False, default='rwkv_vocab_v20230424')
    parser.add_argument('--states_file', type=str, required=False, default='/home/rwkv/Peter/rwkv_graphrag/agents/triples/rwkv6_7b_v2.1_triples.pth')
    parser.add_argument('--token_count', type=int, default=1500, help='Token count for generation')

    args = parser.parse_args()

    process_input(args.input_file, args.output_file, args.model_path, args.vocab_path, args.states_file, args.token_count)

