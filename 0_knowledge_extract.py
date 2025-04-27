import json
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
import xml.etree.ElementTree as ET
import re
import json
from utils.utils import *
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel
import torch
import textwrap
from pathlib import Path
import os

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
BGE_MODEL_DIR = PROJECT_ROOT / "models/bge-m3"
# 确保目录存在
for dir_path in [DATA_DIR, DATASET_DIR, KNOWLEDGE_DIR, BGE_MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def func_1():
    for Field in math_domains:
        train_field_subfield_ids = {}
        train_subfield_ids = {}
        for entry in datas:
            id = entry["id"]
            if entry["field"] == Field:
                key = entry["subfield"]
                # Initialize or update the dictionary entry
                if key not in train_field_subfield_ids:
                    train_field_subfield_ids[key] = [entry["id"]]
                else:
                    train_field_subfield_ids[key].append(entry["id"])

        for i in train_field_subfield_ids:
            train_field_subfield_ids[i] = list(set(train_field_subfield_ids[i]))
            if len(train_field_subfield_ids[i]) > 5:
                train_subfield_ids[i] = train_field_subfield_ids[i]
                print(i, len(train_field_subfield_ids[i]))
        all_nums = []
        for i in train_subfield_ids:
            all_nums += train_subfield_ids[i]
            temp_ids = train_subfield_ids[i]
            levels = [train_dataset[int(i)]["level"] for i in temp_ids]
            # 根据levels对temp_ids从小到大排序
            temp_ids = [x for _, x in sorted(zip(levels, temp_ids))]
            temp_ids = list(set(temp_ids))
            train_subfield_ids[i] = temp_ids

        # 创建Field目录
        field_dir = KNOWLEDGE_DIR / Field
        field_dir.mkdir(parents=True, exist_ok=True)

        # 保存知识解析结果
        dump_json(train_subfield_ids, field_dir / "knowledge_parsed.json")


def func_2():
    for Field in math_domains:
        subfield_cluster_ids = {}
        field_dir = KNOWLEDGE_DIR / Field
        subfield_ids = read_json(field_dir / "knowledge_parsed.json")

        #!遍历每一个subfield,通过keypoints,计算语义相似度
        for subfield in subfield_ids:
            ids = subfield_ids[subfield]
            print("subfield: ", subfield)
            print("size", len(ids))
            if len(ids) < 8:
                cluster_ids = {"0": ids}
                subfield_cluster_ids[subfield] = cluster_ids
                continue

            temp_texts = []
            for id in ids:
                temp = []
                for data in datas:
                    if (
                        data["id"] == id
                        and data["field"] == Field
                        and data["subfield"] == subfield
                    ):
                        temp += data["keypoints"]
                temp_text = (
                    f"Field: {Field}, Subfield: "
                    + subfield
                    + "\nKey Points: "
                    + "; ".join(temp)
                )
                temp_texts.append(temp_text)

            embedding = model.encode(temp_texts, batch_size=12)["dense_vecs"]
            embedding = np.array(embedding)
            similarity_matrix = np.dot(embedding, embedding.T)
            print("similarity_mean: ", np.mean(similarity_matrix))

            #! 根据similarity_matrix进行聚类
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            # Determine the optimal number of clusters
            sil_scores = []
            for k in range(2, min(len(embedding), 15)):
                kmeans = KMeans(n_clusters=k)
                labels = kmeans.fit_predict(embedding)
                sil_score = silhouette_score(embedding, labels)
                sil_scores.append((sil_score, k))

            # Select the number of clusters with the highest silhouette score
            best_k = max(sil_scores, key=lambda x: x[0])[1]
            print("silhouette_scores:", sil_scores)
            print("Best number of clusters:", best_k)

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=best_k)
            labels = kmeans.fit_predict(embedding)

            # Output each cluster size
            cluster_sizes = defaultdict(int)
            for label in labels:
                cluster_sizes[label] += 1
            print("Cluster sizes:", cluster_sizes)
            print("Silhouette Score:", max(sil_scores, key=lambda x: x[0])[0])
            print("\n\n")

            cluster_ids = {}
            for i in range(best_k):
                cluster_ids[i] = []
            for i in range(len(ids)):
                cluster_ids[labels[i]].append(ids[i])
            subfield_cluster_ids[subfield] = cluster_ids

        # 保存聚类结果
        dump_json(subfield_cluster_ids, field_dir / "subfield_cluster_ids.json")


def func_3():
    # Todo: 构造inital_tool_make_query
    subfield_cluster_querys = []
    for Field in math_domains:
        field_dir = KNOWLEDGE_DIR / Field
        subfield_cluster_ids = read_json(field_dir / "subfield_cluster_ids.json")

        for subfield in subfield_cluster_ids:
            print(subfield)
            for cluster in subfield_cluster_ids[subfield]:
                ids = subfield_cluster_ids[subfield][cluster]
                #! 如果ids的长度大于50，随机抽取50个，避免生成的query过长
                if len(ids) > 50:
                    random.sample(ids, 50)
                print("cluster:", cluster, "size:", len(ids))
                examples_prompt = "---\nHere is the batch of tasks.\n"
                for id in ids:
                    key_points = ""
                    for data in datas:
                        if (
                            data["id"] == id
                            and data["field"] == Field
                            and data["subfield"] == subfield
                        ):
                            key_points = data["keypoints"]
                    examples_prompt += (
                        "### Task\n" + "\nKey Points: " + "; ".join(key_points) + "\n\n"
                    )
                prompt = """[Temperature = 0.3]
**Act like a seasoned Python developer and mathematician with extensive experience in crafting math tools and functions. Your objective is to develop reusable Python functions that can be helpful for as many mathematical problems as possible **

To the knowledge point `{}' in the field of `{}', there is a batch of similar subtasks below. Please generate one or more general tool to solve as many subtasks as possible about this knowledge point.
    - Design and implement Python functions to solve the subtasks. Use appropriate Python libraries such as `sympy`, `numpy`, `scipy`, and `math` and you should import inside the function.
    - Make sure that each function is fully functional and suitable for a variety of different situations, just like the API for a real-world scenario.
    - Abstract specific literals and constants from the problem's context, replacing them with variables and parameters to enhance the tools' applicability across different scenarios.
    - Function and variable names are as descriptive as possible.
    - Document each function extensively using the NumPy docstring format to explain the task description, solution description, parameters, return values, and examples.

Output Format:
### Thought
Analyze a given math field and knowledge point, look at the subtasks, then consider how many math tools you need to design, what is their tasks, and so on.

### Math tools 1
```python
```
### Math tools 2 (if needed)
```python
```
        """.format(
                    subfield, Field
                )
                query = prompt + "\n\n" + examples_prompt
                subfield_cluster_querys.append(
                    {
                        "id": "math" + "@" + Field + "@" + subfield + "@" + cluster,
                        "query": query,
                    }
                )
    subfield_cluster_querys = subfield_cluster_querys * 3
    save_jsonl(
        subfield_cluster_querys,
        KNOWLEDGE_DIR / "subfield_cluster_inital_tool_make_querys_all.jsonl",
    )


if __name__ == "__main__":
    # 加载数据
    train_dataset = load_jsonl(DATASET_DIR / "math/train.jsonl")
    math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")
    datas = load_jsonl(KNOWLEDGE_DIR / "train_knowledge_parsed.jsonl")

    # 初始化模型
    model = BGEM3FlagModel(str(BGE_MODEL_DIR), use_fp16=True, device="cuda:7")

    func_3()
