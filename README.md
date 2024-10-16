# 功能介绍
rwkv-kg是一款基于rwkv模型的知识图谱提取工具。用户输入300-500token的原文，可以得到标准json格式的命名体和命名体之间的关系组成的知识图谱。


# 模型下载

* 请在：https://huggingface.co/BlinkDL/rwkv-6-world/tree/main 下载RWKV6_7B模型。

* 请在：https://huggingface.co/collections/yueyulin/rwkvgraphragstates-66f3cc889240b4f1d2ec4ae1 下载graphrag所需要的states

* 请在 https://github.com/Ojiyumm/rwkv_kg 的agents4rwkvgraph中找到所有states的说明以及测试代码。 在src中找到提取知识图谱的代码样例。


# 使用方式

寻找实体种类

* input_text： 需要生成图谱的文本 
* output_path： 实体种类储存地址 
* base_model： 模型地址 
* persona_domain_state： persona_states 地址 
* entity_types_state： entity_types地址

``` python

python entity_types.py --input_text  --output_path /home/rwkv/Peter/rwkv_graphrag/data/triple_input.jsonl' --base_model /home/rwkv/Peter/models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth --persona_domain_state /home/rwkv/Peter/rwkv_graphrag/agents/persona_domain_states/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth --entity_types_state /home/rwkv/Peter/rwkv_graphrag/agents/entity_type_extraction/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth

```

提取三元组

* input_file： 用来提取知识图谱的输入，和entity_types.py的输出相同
* output_file： 知识图谱储存地址
* model path： 模型地址
* states_file： triples_extraction states地址

``` python
python triple_extractions.py --input_file /home/rwkv/Peter/rwkv_graphrag/data/triple_input.jsonl --output_file /home/rwkv/Peter/rwkv_graphrag/data/triples.jsonl --model_path /home/rwkv/Peter/models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth --states_file /home/rwkv/Peter/rwkv_graphrag/agents/triples/rwkv6_7b_v2.1_triples.pth

```

# 例子
```
输入： 有个空空道人访道求仙，从大荒山无稽崖青埂峰下经过，忽见一大块石上字迹分明，编述历历，《石头记》是也。空空道人将《石头记》抄录下来，改名为《情僧录》。至吴玉峰题曰《红楼梦》。东鲁孔梅溪则题曰《风月宝鉴》。后因曹雪芹于悼红轩中披阅十载，增删五次，纂成目录，分出章回，则题曰《金陵十二钗》。姑苏乡宦甄士隐梦见一僧一道携无缘补天之石（通灵宝玉）下凡历练，又讲绛珠仙子为报神瑛侍者浇灌之恩追随神瑛侍者下世为人，以泪报恩。梦醒后，抱女儿英莲去看“过会”[2]。甄士隐结交并接济了寄居于隔壁葫芦庙内的胡州人氏贾化（号雨村）。某日，贾雨村造访甄士隐，无意中遇见甄家丫鬟娇杏，以为娇杏对其有意。中秋时节，甄士隐于家中宴请贾雨村，得知贾雨村的抱负后，赠银送衣以作贾雨村上京赴考之盘缠，第二天，贾雨村不辞而别便上路赴考。第二年元宵佳节当晚，甄家仆人霍启在看社火花灯时，不慎丢失了甄士隐唯一的女儿英莲[3]。三月十五日，葫芦庙失火祸及甄家，落魄的甄士隐带家人寄居于如州岳丈封肃家中，后遇一僧一道，悟出《好了歌》真谛，随僧道而去。
```
```
输出：{"triples": " {"entities": [{"entity": "空空道人", "description": "一位在大荒山无稽崖青埂峰下经过的访道求仙者，他从石头记中得到启示，将其改名为情僧录。", "id": 1}, {"entity": "石头记", "description": "《石头记》是一部由曹雪芹编写的小说，讲述了红楼梦的故事。", "id": 2}, {"entity": "情僧录", "description": "《情僧录》是空空道人对《石头记》的改编，将其重新命名为《情僧录》。", "id": 3}, {"entity": "红楼梦", "description": "《红楼梦》是曹雪芹所著的一部长篇小说，被认为是中国古代小说的巅峰之作。", "id": 4}, {"entity": "金陵十二钗", "description": "金陵十二钗是《红楼梦》中的一个角色群体，包括贾宝玉、林黛玉、薛宝钗等人物。", "id": 5}, {"entity": "甄士隐", "description": "甄士隐是《红楼梦》中的一个角色，他在葫芦庙失火时失去了女儿英莲。", "id": 6}, {"entity": "贾雨村", "description": "贾雨村是《红楼梦》中的一个角色，他与甄士隐有过交往，并赠送了银子给甄家以帮助甄家度过难关。", "id": 7}, {"entity": "曹雪芹", "description": "曹雪芹是《红楼梦》的作者，他在书中描绘了丰富的人物形象和复杂的社会关系。", "id": 8}, {"entity": "封肃家", "description": "封肃家是曹雪芹居住的地方，也是《红楼梦》故事发生的背景之一。", "id": 9}, {"entity": "如州岳丈封肃家", "description": "如州岳丈封肃家是曹雪芹居住的地方，也是《红楼梦》故事发生的背景之一。", "id": 10}, {"entity": "葫芦庙失火", "description": "葫芦庙失火是《红楼梦》中的一个重要事件，导致了甄士隐女儿英莲的离世。", "id": 11}], "relationships": [{"source": "空空道人", "target": "石头记", "relationship": "空空道人从石头记中得到启示，将其改名为情僧录。", "id": 12}, {"source": "空空道人", "target": "情僧录", "relationship": "空空道人将《石头记》改名为情僧录。", "id": 13}, {"source": "甄士隐", "target": "贾雨村", "relationship": "甄士隐在《红楼梦》中与贾雨村有过交往，并赠送了银子给贾雨村以帮助贾雨村上京赴考。", "id": 14}, {"source": "甄士隐", "target": "曹雪芹", "relationship": "甄士隐在《红楼梦》中与曹雪芹有过交往，并赠送了银子给曹雪芹以帮助曹雪芹上京赴考。", "id": 15}, {"source": "甄士隐", "target": "封肃家", "relationship": "甄士隐在《红楼梦》中与封肃家有过交往，并寄居于封肃家中。", "id": 16}, {"source": "甄士隐", "target": "如州岳丈封肃家", "relationship": "甄士隐在《红楼梦》中与如州岳丈封肃家有过交往，并寄居于如州岳丈封肃家中。", "id": 17}]}"}

# 图谱可视化马上加上
