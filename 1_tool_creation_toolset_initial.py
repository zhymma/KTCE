import json
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from math_normalize import *
from pathlib import Path
import os

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
TOOLSET_DIR = DATA_DIR / "toolset"
INITIAL_TOOLSET_DIR = TOOLSET_DIR / "initial"
OPTIMIZE_TOOLSET_DIR = TOOLSET_DIR / "optimize"
BGE_MODEL_DIR = DATA_DIR / "bge_model"
# 确保目录存在
for dir_path in [
    DATA_DIR,
    DATASET_DIR,
    KNOWLEDGE_DIR,
    TOOLSET_DIR,
    INITIAL_TOOLSET_DIR,
    OPTIMIZE_TOOLSET_DIR,
    BGE_MODEL_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)


def func_1(Field, subfield_cluster_ids, subfield_cluster_tool_make):
    """对每个subfield的所有cluster进行去重"""
    subfield_tools = {}
    model = BGEM3FlagModel(str(BGE_MODEL_DIR), use_fp16=True, device="cuda:7")

    for _subfield in subfield_cluster_ids:
        toolset = []
        for line in subfield_cluster_tool_make:
            id = line["id"]
            if "math" not in id:
                continue
            field = id.split("@")[1]
            subfield = id.split("@")[2]
            cluster = id.split("@")[3]
            if field != Field:
                continue
            if _subfield == subfield:
                if "answer_mode4" not in line:
                    continue
                response = line["answer_mode4"]
                response_codes = extract_program(response)
                tool_functions = extract_math_tools(response_codes)
                toolset.extend(tool_functions)

        print("subfield: ", _subfield)
        print("size: ", len(toolset))

        # 对之前的subfield进行去重
        sentence_pairs = [[i, j] for i in toolset for j in toolset]
        similarity_matrix = model.compute_score(
            sentence_pairs, batch_size=12, weights_for_different_modes=[0.4, 0.2, 0.4]
        )["colbert+sparse+dense"]
        similarity_matrix = np.array(similarity_matrix).reshape(
            len(toolset), len(toolset)
        )

        # 使用聚类算法分组
        distance_threshold = 1 - 0.80
        distance_matrix = 1 - similarity_matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        ).fit(distance_matrix)
        print(len(set(clustering.labels_)))

        # 收集每个cluster的tools
        clusters = {i: [] for i in range(len(set(clustering.labels_)))}
        for tool, label in zip(toolset, list(clustering.labels_)):
            clusters[label].append(tool)

        # 将聚类结果保存到字典中
        subfield_tools[_subfield] = [clusters[i] for i in clusters]

    # 保存结果
    output_file = INITIAL_TOOLSET_DIR / Field / "subfield_tools.json"
    dump_json(subfield_tools, output_file)


def func_2(Field):
    """生成初始工具集"""
    output_file = INITIAL_TOOLSET_DIR / Field / "subfield_initial_tools.json"
    if output_file.exists():
        print("subfield_initial_tools.json exists")
        return

    subfield_tools = read_json(INITIAL_TOOLSET_DIR / Field / "subfield_tools.json")
    subfield_cluster_ids = read_json(
        KNOWLEDGE_DIR / Field / "subfield_cluster_ids.json"
    )

    # 使用chatgpt从subfield_tools中去除重复的工具，通过比较调用工具的输出
    subfield_initial_tool = {}
    validation_template_multi_1 = """Here are several math tools with similar functionalities. Your task is to generate the same example function calling for all tool, and then compare the output of the tools, the no bug tool and have the same output as the right tool. First, you should output the function calling arg values in nutural language (Describes an instance of a tool calling), then we will use it to generate function calling next. The output should contain the values ​​of  parameters or variables.

### Input ###
The tools are:
```python
{}
```

### Output:
"""
    validation_template_multi_2 = """Here are a math tool with an example of function calling. Your task is to generate the  function calling by python. Please output the function calling code in block ```python ```. The last line print the result of the function calling. Note that all variables or parameters must be defined and initialized before calling. Note that you should not regenerate the function code, just function calling.

### Input ###
The tool is:
```python
{}
```
The example function calling is:
{}
### Output:
"""

    validation_template_single = """Here are a math tool by python function or class. Your task is to generate  example function calling to validate corectness the tool. Please output the function calling code in block ```python ```. The last line print the result of the function calling. Note that all variables or parameters must be defined and initialized before calling. Note that you should not regenerate the function code, just function calling.

### Input ###
The tools are:
```python
{}
```

### Output ###
```python
# Example function call
```
"""

    for subfield in subfield_tools:
        ids = []
        for cluster in subfield_cluster_ids[subfield]:
            ids += subfield_cluster_ids[subfield][cluster]
        cluster_sizes = [len(i) for i in subfield_tools[subfield]]
        sorted_indices = np.argsort(cluster_sizes)[::-1]  # 降序排列
        best_tools_in_subfield = []
        for idx in sorted_indices:
            cluster_tools = subfield_tools[subfield][idx]
            if len(cluster_tools) > 8:
                cluster_tools = random.sample(cluster_tools, 8)
            print("subfield: ", subfield)
            print("cluster size: ", len(cluster_tools))
            tool_string = ""
            for idx, t in enumerate(cluster_tools):
                tool_string += f"# Tool {idx}:\n{t}\n\n"
            tool_callings = []
            if len(cluster_tools) == 1:
                prompt = validation_template_single.format(tool_string)
                result = chat_api(
                    "gpt-3.5-turbo-0125", prompt, 0, system_msg, temperature=0.3
                )
                code = extract_program(result)
                generate_tools = extract_math_tools(code)
                for g_ts in generate_tools:
                    code = code.replace(g_ts, "")
                if ("def") in code:
                    print("def in code error: ", code)
                    code += "ads\nwd()"
                tool_callings.append(code)
            else:
                prompt = validation_template_multi_1.format(tool_string)
                system_msg = "You are an experienced Python developers and mathematicians with extensive experience can call the math tools."
                example_calling = chat_api(
                    "gpt-3.5-turbo-0125", prompt, 0, system_msg, temperature=0.3
                )
                for idx, t in enumerate(cluster_tools):
                    prompt = validation_template_multi_2.format(
                        f"# Tool {idx}:\n{t}\n\n", example_calling
                    )
                    result = chat_api(
                        "gpt-3.5-turbo-0125", prompt, 0, system_msg, temperature=0.3
                    )
                    code = extract_program(result)
                    generate_tools = extract_math_tools(code)
                    for g_ts in generate_tools:
                        code = code.replace(g_ts, "")
                    if ("def") in code:
                        print("def in code error: ", code)
                        code += "ads\nwd()"
                    tool_callings.append(code)

            if len(tool_callings) == len(cluster_tools):
                # 找到对应的tool和tool calling
                codes = [i + "\n" + j for i, j in zip(cluster_tools, tool_callings)]
                reports = []
                results = []
                for code in codes:
                    report, result = execute_code("", code)
                    print(report, result)
                    if report:
                        result = normalize_answer(result)
                    reports.append(report)
                    results.append(result)
                # 找到正确的工具
                if len(cluster_tools) == 1:
                    if reports[0] == True or results[0] == 0:
                        best_tools_in_subfield.append(cluster_tools[0])
                else:
                    # 对于report为true，若只有一个为true这为正确工具，若有多个为true，根据结果是否一致，结果数量最多的对应的tool为正确的工具
                    valid_reports = [i for i, r in enumerate(reports) if r == True]
                    if len(valid_reports) == 0:
                        selected_tool_idx = -1
                    elif len(valid_reports) == 1:
                        selected_tool_idx = valid_reports[0]
                    else:
                        selected_tool_idx = -1
                        result_count = {}
                        for i in valid_reports:
                            res = results[i]
                            if res in result_count:
                                result_count[res].append(i)
                            else:
                                result_count[res] = [i]
                        max_count = 0
                        selected_tool_idxs = []
                        for res, idxs in result_count.items():
                            if len(idxs) > max_count:
                                max_count = len(idxs)
                                selected_tool_idxs = idxs

                        if len(selected_tool_idxs) == 1:
                            selected_tool_idx = selected_tool_idxs[0]
                        else:
                            # 选择对应的tool的长度最短的
                            min_len = 1000000
                            for idx in selected_tool_idxs:
                                if len(cluster_tools[idx]) < min_len:
                                    min_len = len(cluster_tools[idx])
                                    selected_tool_idx = idx
                        if selected_tool_idx != -1:
                            best_tools_in_subfield.append(
                                cluster_tools[selected_tool_idx]
                            )
            else:
                print("error: ", len(tool_callings), len(cluster_tools))
        subfield_initial_tool[subfield] = best_tools_in_subfield
        print(
            "Field: ",
            Field,
            "subfield: ",
            subfield,
            "len_tools: ",
            len(best_tools_in_subfield),
        )

    # 保存结果
    dump_json(subfield_initial_tool, output_file)


def func_3(Field, subfield_initial_tools):
    """生成tools_iter_0"""
    subfield_tools_iter_0 = {}
    for subfield in subfield_initial_tools:
        tools = subfield_initial_tools[subfield]
        subfield_tools_iter_0[subfield] = []
        for tool in tools:
            tool = tool.strip()
            tool_type = ""
            if tool.startswith("def "):
                tool_type = "function"
            elif tool.startswith("class "):
                tool_type = "class"
            if tool_type == "function":
                description = extract_function_description(tool)
                docstring = extract_function_docstring(tool)
                tool_name = extract_function_name(tool)
                tool = remove_function_docstring(tool)
            elif tool_type == "class":
                description = extract_class_description(tool)
                docstring = extract_class_docstring(tool)
                tool_name = extract_class_name(tool)
            tool_json = {
                "tool": tool,
                "subfield": subfield,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "description": description,
                "docstring": docstring,
                "Freq": 0,
                "TSR": 0,
            }
            if subfield in subfield_tools_iter_0:
                subfield_tools_iter_0[subfield].append(tool_json)
            else:
                subfield_tools_iter_0[subfield] = [tool_json]

    # 保存结果
    output_file = OPTIMIZE_TOOLSET_DIR / Field / "subfield_tools_iter_0.json"
    dump_json(subfield_tools_iter_0, output_file)


def main():
    """主函数"""
    # 加载数据
    subfield_cluster_tool_make = load_jsonl(
        KNOWLEDGE_DIR / "math_tabmwp_subfield_cluster_inital_tool_make_querys_all.json"
    )
    math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")

    # 创建必要的目录
    for Field in math_domains:
        (INITIAL_TOOLSET_DIR / Field).mkdir(parents=True, exist_ok=True)
        (OPTIMIZE_TOOLSET_DIR / Field).mkdir(parents=True, exist_ok=True)

    # 处理每个领域
    for Field in math_domains:
        subfield_tools_file = INITIAL_TOOLSET_DIR / Field / "subfield_tools.json"
        if not subfield_tools_file.exists():
            subfield_cluster_ids = read_json(
                KNOWLEDGE_DIR / Field / "subfield_cluster_ids.json"
            )
            func_1(Field, subfield_cluster_ids, subfield_cluster_tool_make)

    # 并行处理初始工具集
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(func_2, Field) for Field in math_domains]
        for future in futures:
            future.result()

    # 生成迭代0的工具集
    for Field in math_domains:
        subfield_initial_tools = read_json(
            INITIAL_TOOLSET_DIR / Field / "subfield_initial_tools.json"
        )
        func_3(Field, subfield_initial_tools)


if __name__ == "__main__":
    main()
