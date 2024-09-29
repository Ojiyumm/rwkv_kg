from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch
import json

def generate_triples(input_text):
    model_path = '/home/rwkv/Peter/model/base/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
    vocab_path = "rwkv_vocab_v20230424"
    instruction ='根据input中的input和entity_types，帮助用户找到文本中的entity_types的实体,标明实体类型并且简单描述，。然后给找到实体之间的关系，并且描述这段关系以及对关系强度打分。 避免使用诸如“其他”或“未知”的通用实体类型。 非常重要的是：不要生成冗余或重叠的实体类型和关系。用JSON格式输出。'
    states_file = '/home/rwkv/Peter/rwkv_graphrag/agents/triples/rwkv6_7b_v2.1_triples.pth'
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
    ctx = f'Instruction: {instruction}\nInput: {input_text}\n\nResponse:'
    print(ctx)
    # Define the print callback function
    

    # Set up the generation arguments
    args = PIPELINE_ARGS(
        temperature=0.8,
        top_p=0.1,
        top_k=0,
        alpha_frequency=0.40,
        alpha_presence=0.31,
        alpha_decay=0.996,
        token_ban=[0],
        token_stop=[0,1,261],
        chunk_len=256
    )
    def my_print(s):
        print(s, end='', flush=True)
    # Generate the response
    triples=pipeline.generate(ctx, token_count=1500, args=args,callback=my_print,state=states_value)
    return triples


with open ('/home/rwkv/Peter/rwkv_graphrag/data/triple_input.jsonl','r',encoding='utf-8') as f:
    for line in f:
            # Load each JSON object
            input = json.loads(line)
            torch.cuda.empty_cache()
            triples=generate_triples(input)
            triples=triples.replace("'", '"')
            formated={"triples":triples}
            
            with open('/home/rwkv/Peter/rwkv_graphrag/data/triples.jsonl','a',encoding='utf-8') as f:
                
                json.dump(formated, f, ensure_ascii=False)
                f.write('\n') 
    

# if __name__ == "__main__":
#     model_path = '/home/rwkv/Peter/model/base/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
#     states_file = '/home/rwkv/Peter/rwkv_graphrag/agents/triples/rwkv6_7b_v2.1_triples.pth'
#     input_text = '{\"input\": \"有个空空道人访道求仙，从大荒山无稽崖青埂峰下经过，忽见一大块石上字迹分明，编述历历，《石头记》是也。空空道人将《石头记》抄录下来，改名为《情僧录》。至吴玉峰题曰《红楼梦》。东鲁孔梅溪则题曰《风月宝鉴》。后因曹雪芹于悼红轩中披阅十载，增删五次，纂成目录，分出章回，则题曰《金陵十二钗》。姑苏乡宦甄士隐梦见一僧一道携无缘补天之石（通灵宝玉）下凡历练，又讲绛珠仙子为报神瑛侍者浇灌之恩追随神瑛侍者下世为人，以泪报恩。梦醒后，抱女儿英莲去看“过会”[2]。甄士隐结交并接济了寄居于隔壁葫芦庙内的胡州人氏贾化（号雨村）。某日，贾雨村造访甄士隐，无意中遇见甄家丫鬟娇杏，以为娇杏对其有意。中秋时节，甄士隐于家中宴请贾雨村，得知贾雨村的抱负后，赠银送衣以作贾雨村上京赴考之盘缠，第二天，贾雨村不辞而别便上路赴考。第二年元宵佳节当晚，甄家仆人霍启在看社火花灯时，不慎丢失了甄士隐唯一的女儿英莲。三月十五日，葫芦庙失火祸及甄家，落魄的甄士隐带家人寄居于如州岳丈封肃家中，后遇一僧一道，悟出《好了歌》真谛，随僧道而去。\"}, {\"entity_types\": [\"文学与神话\", \"历史背景\", \"影响分析\", \"改编过程\", \"角色贡献\", \"写作技巧评估\"]}'
    
#     triples=generate_triples(model_path, states_file, input_text)
#     print(triples)


# def create_graph(demo_data):
#     import igraph as ig
#     import leidenalg as la
#     demo_data=json.loads(demo_data)
#     entities = demo_data['entities']
#     relations = demo_data['relationships']

#     edge_relations = {}
#     nodes = set([])
#     for entity in entities:
#         nodes.add(entity['entity'])

#     for relation in relations:
#         head = relation.get('source')
#         tail = relation.get('target')
#         if head not in nodes or tail not in nodes:
#             continue
#         relation_label = relation.get('relationship', '')
#         # nodes.add(head)
#         # nodes.add(tail)
#         if (head, tail) not in edge_relations:
#             edge_relations[(head, tail)] = relation_label
#     edges = list(edge_relations.keys())
#     edge_relations = list(edge_relations.values())
#     # 创建一个图
#     g = ig.Graph(directed=False)
#     # 添加节点
#     vertices = list(nodes)
#     vertices_id_map = {item: i for i, item in enumerate(vertices)}
#     g.add_vertices(vertices)

#     # 添加边
#     edges = [(vertices_id_map[edge[0]], vertices_id_map[edge[1]]) for edge in edges]
#     print(len(edges))
#     g.add_edges(edges)

#     # 允许的最大社区大小
#     _max_comm_size = 3

#     # 使用 Leiden 算法进行社区检测
#     partition = la.find_partition(g, partition_type=la.ModularityVertexPartition, max_comm_size=_max_comm_size,
#                                   seed=5)
#     # 计算节点的坐标
#     pos = g.layout("kk")
#     # 绘制图形
#     visual_style = {
#         'vertex_size': 30,
#         'vertex_label': g.vs['name'],#[random.randint(0,100) for i in g.vs['name']],  # 显示节点标签
#         'vertex_label_dist': 1.5,  # 节点标签与节点之间的距离
#         'edge_width': 1,
#         'layout': pos,
#         'bbox': (1000, 1000),  # 图像尺寸
#         'margin': 70,  # 边距
#         'edge_label_dist': 0.2,  # 边标签与边之间的距离
#         'vertex_label_font' : 'WenQuanYi Zen Hei',
#         'font': 'WenQuanYi Zen Hei'
#     }
#     if edge_relations:
#         visual_style['edge_label'] = edge_relations
#         visual_style["edge_label_color"] = 'darkgray'  # 边标签的颜
#         visual_style["edge_label_pos"] = 1
#     ig.plot(partition, "test1.png", **visual_style)

# create_graph(triples)    