import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from math_normalize import *
import copy
import argparse
from pathlib import Path
import os

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
TOOLSET_DIR = DATA_DIR / "toolset"
OPTIMIZE_TOOLSET_DIR = TOOLSET_DIR / "optimize"
CODE_EXEC_DIR = PROJECT_ROOT / "code_exec"
BGE_MODEL_DIR = DATA_DIR / "bge_model"

# 确保目录存在
for dir_path in [
    DATA_DIR,
    DATASET_DIR,
    KNOWLEDGE_DIR,
    TOOLSET_DIR,
    OPTIMIZE_TOOLSET_DIR,
    CODE_EXEC_DIR,
    BGE_MODEL_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 提示模板定义
get_tool_prompt_template = """As a Python programming and math expert, given a math question and some math tools, please decide tools can be used to solve this question. Do not solve this question, just judge which tools can be used to solve this question.

### Format ###
Tools are listed below:

No. 0:
Tool 0

No. 1:
Tool 1

...

No. N:
Tool N

Here are some instructions you should follow:
- Analyse what subtasks are in the problem.
- Deeply understand all the tools provided, refer to their function name and code.
- Please judge which tools (one or more) can be used to solve this problem, and give your thoughts.
- If there are tools useful, please output <answer> </answer> with the numeric number list, e.g.: <answer> [N1,N2] </answer>
- If there are not tools useful, please output <answer>[]</answer>
- Take a deep breath
- Think step by step 
- I will tip $200

### Question:
{}

Here are the math tools in the field {}, subfield {}:
{}

"""
tpot_prompt_template = '''As a Python programming and math expert, given a math question and math tools, please use the math tools to solve the math question by python program. Note that do not regenerate the tool code and function, just use the tool code and function to solve the question.

Here are some instructions you should follow:
- You need to understand the tool function in depth to ensure that the parameters called are correct. Directly call tool function without external imports, do not modify the tool function.
- You can also use python libraries, like sympy, math, scipy, numpy, etc. or other packages if necessary.
- Please pay attention to the conditions of the question, conduct accurate logical reasoning, and obtain results that meet the requirements.
---

### Math Tools:
```python
{tool_string}
```
---

Here are some examples you can refer to, you should call the tool functions directly in your code.

{examples_string}

# Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
```python
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
print(solution())
```
---
Now it's your turn!
- Take a deep breath
- Think step by step 
- I will tip $200

# Question: {question}
# Solution: {solution}
'''

delete_tool_prompt_template = """As a Python programming and math expert, you are given a set of tools and their usage statistics. Please decide if any tool should be removed, Only remove the function that you think is not needed anymore in future tasks.

### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Tools and Usage Statistics:
{tools_stats}

### Instructions:
- Analyze the tools and their usage statistics.
- Consider the tool's usage frequency and the tool's generality and versatility.
- Don't remove more than 5 tools to maintain diversity in your toolset.
- If a tool should be removed, provide the tool index in the answer, e.g.: <answer> [N1,N2] </answer>

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Remove (at most 3 tools):
<answer> [Tool Indexes] </answer>
"""

delete_tool_prompt_template_with_wrong_info = """As a Python programming and math expert, you are given a set of tools and their usage statistics. Please decide if any tool should be removed, Only remove the function that you think is not needed anymore in future tasks.

### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Tools and Usage Statistics:
{tools_stats}

### Previous Errors:
- Previously, some tools were incorrectly removed, which led to a decrease in the Toolset Coverage on Given Math Dataset and Accuracy of Using the Toolset on Given Math Dataset in Python Inference. It is crucial to avoid such mistakes to maintain or improve these metrics.
- Here are the tools that were incorrectly removed in the previous iteration:
{wrong_info}
- Here are the Toolset Information after the wrong action:
Toolset Coverage on Given Math Dataset: {wrong_TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {wrong_TA}

### Instructions:
- Analyze the tools and their usage statistics.
- Consider the tool's usage frequency and the tool's generality and versatility.
- Don't remove more than 5 tools to maintain diversity in your toolset.
- If a tool should be removed, provide the tool index in the answer, e.g.: <answer> [N1,N2] </answer>
- Don't make previous mistakes that reduced the coverage and accuracy of your toolset.

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Remove (at most 5 tools):
<answer> [Tool Indexes] </answer>
"""

add_tool_prompt_template = """As a Python programming and math expert, you are given a set of tools, their usage statistics, and uncovered problems by the toolset. Please decide if any new tool should be added to improve the coverage of the toolset and solve the task in the uncovered problems. Provide your reasoning for each decision, if need to add, please output the new tool code with docstring, Note that it must a general tool for many problems and running accurately. 


### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Toolset:
{tools_stats}

### Uncovered Problems:
{unsolved_problems}

### Instructions:
- Analyze and the uncoverd problems and consider what task in this subfield are not solved in current toolset.
- If the task is not solved by the provided toolset, the tool should be added to solve this task. The new tool should be different from the existing tools in the toolset.
- If a tool should be added, please generate the tool code with docstring. As an extension, you can use any tool in the previous toolset in new tool. You need to write these sub-tools as sub-functions completely inside the new tool to ensure that the new tool can run accurately.
- The added function should be general enough to be used in future tasks. For instance, if you encounter a problem that this function can solve, or one step of it, you can use the generated function directly instead of starting from scratch 
- The added new function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable. 
- Replace specific strings or variable names with general variables to enhance the tool's applicability to various queries. All names used inside the function should be passed in as arguments.

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Add New Tool (one or more):
```python
[New Tool Code] 
```
"""

add_tool_prompt_template_with_wrong_info = """As a Python programming and math expert, you are given a set of tools, their usage statistics, and uncovered problems by the toolset. Please decide if any new tool should be added to improve the coverage of the toolset and solve the task in the uncovered problems. Provide your reasoning for each decision, if need to add, please output the new tool code with docstring, Note that it must a general tool for many problems and running accurately. 


### Toolset Information:
Field: {field}
Subfield: {subfield}
Toolset Size: {n}
Toolset Coverage on Given Math Dataset: {TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {TA}

### Toolset:
{tools_stats}

### Uncovered Problems:
{unsolved_problems}

### Previous Errors:
- Previously, some tools were incorrectly added, which led to a decrease in the Toolset Coverage on Given Math Dataset and Accuracy of Using the Toolset on Given Math Dataset in Python Inference. It is crucial to avoid such mistakes to maintain or improve these metrics.
- Here are the tools that were incorrectly added in the previous iteration:
{wrong_info}
- Here are the Toolset Information after the wrong action:
Toolset Coverage on Given Math Dataset: {wrong_TSC}
Accuracy of Using the Toolset on Given Math Dataset in Python Inference: {wrong_TA}


### Instructions:
- Analyze and the uncoverd problems and consider what task in this subfield are not solved in current toolset.
- If the task is not solved by the provided toolset, the tool should be added to solve this task. The new tool should be different from the existing tools in the toolset.
- Don't make previous mistakes that reduced the coverage and accuracy of your toolset.
- ** If a tool should be added, please generate the tool code with docstring. As an extension, you can use any tool in the previous toolset in new tool. You need to write these sub-tools as sub-functions completely inside the new tool to ensure that the new tool can run accurately. **
- The added function should be general enough to be used in future tasks. For instance, if you encounter a problem that this function can solve, or one step of it, you can use the generated function directly instead of starting from scratch 
- The added new function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable. 
- Replace specific strings or variable names with general variables to enhance the tool's applicability to various queries. All names used inside the function should be passed in as arguments.

Please provide your decision in the following format:
Reasoning: [Your reasoning]
Add New Tool (one or more):
```python
[New Tool Code] 
```
"""

modify_tool_prompt_template = """As a Python programming and math expert, you are given a math tool and its usage statistics. Please decide if this tool should be evolved to improve its accuracy, flexibility, and ease of use. If evolution is needed, please output the new tool code with a docstring. * Note that the new tool is an evolution of the original tool, and there function name and code must be similar and running accurately !!! *


### Tool Information and Usage Statistics:
# Field: {field}
# Subfield: {subfield}
# Tool Usage Frequency: {Freq}
# Tool Success Rate: {TSR}%
# Tool docstring : {docstring}
# Tool code:
```python
{tool}
```
# Wrong tool calings:
{wrong_tool_callings}
# Tool usage experience:
{experience_pool}

---

### Instructions:
- Evolution often includes expanding the tool's functionality, handling different scenarios, changing parameters and return values, and improving ease of use.
- Pay more attention to failed tasks and corresponding error information. If necessary, optimize the features used in these tasks based on conversation history.
- Function calls may fail due to incorrect input parameters (missing parameters) or incorrect function code implementation. You should focus more on the implementation of the function code and make function calls easier to succeed.
- Based on conversation history, do not modify functions that are effective and crucial for solving the problem. Modify only when necessary.
- A low success rate may be due to the difficulty of the problem, not tool usage errors. You need to judge based on the output content. If the tool is not the cause of the error, you should not modify the tool and update the experience pool instead.

* If the tool can be evolved, and provide your reasoning and the new tool code with docstring and try to update/generate the **experience pool** to prevent similar errors and guide LLM to use the tool accurately. Note what modified is the tool itself, not the wrong calling code of the tool. *

Output format 1 (evolve the tool):
Reasoning: [Your reasoning]
Evolved tool code:
```python
[new tool code] 
```
Experience pool:
<experience_list>
experience content
</experience_list>

---

* If the tool has no problem, consider to update/generate the **experience pool** to prevent similar errors and guide LLM to use the tool accurately. *

Output format 2 (not evolve the tool):
Reasoning: [Your reasoning]
Experience pool:
<experience_list>
experience content
</experience_list>
"""


def get_kps(id, Field, subfield):
    """获取知识点的关键信息"""
    global parsed_knowledge
    for line in parsed_knowledge:
        if line["id"] == id and line["field"] == Field and line["subfield"] == subfield:
            return line["keypoints"]
    return []


def get_fields_subfields(id):
    """获取字段和子字段信息"""
    global parsed_knowledge
    fields = []
    subfield = []
    for line in parsed_knowledge:
        if line["id"] == id:
            fields.append(line["field"])
            subfield.append(line["subfield"])
    return fields, subfield


def get_useful_tools(response):
    """从响应中提取有用的工具"""
    called_idx = []
    try:
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, response)
        for match in matches:
            ans = eval(match)
            for idx in ans:
                idx = int(idx)
                called_idx.append(idx)
    except Exception as e:
        print("error: ", e)
        print("response: ", response)
    return called_idx


def extract_experience_pool(response):
    """从响应中提取经验池信息"""
    experience_pool = ""
    try:
        pattern = r"<experience_list>(.*?)</experience_list>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            experience_pool = match.group(1).strip()
    except Exception as e:
        print("error: ", e)
        print("response: ", response)
    return experience_pool


def calculate_loss(Q_tool_values, Q_set, alpha, beta, gamma, n, k):
    """计算损失值"""
    Q_tool = Q_tool_values.sum()
    loss = alpha * Q_tool + beta * Q_set + gamma * max(0, n - k)
    return loss


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


def valid_tool(tool):
    """验证工具的有效性"""
    prompt = validation_template_single.format(tool)
    response = chat_api("gpt-3.5-turbo", prompt, 0, "", temperature=0.0)
    code = extract_program(response)
    generate_tools = extract_math_tools(code)
    for g_ts in generate_tools:
        code = code.replace(g_ts, "")
    if "def" in code or "class" in code:
        print("def or class in code error: ", code)
    code = tool + "\n" + code
    report, result = execute_code("", code)
    if not report:
        return False
    return True


def get_tool_json(tool):
    """获取工具的JSON表示"""
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
    return tool_json


def process_problem(
    id,
    tool_set_string,
    model,
    tool_jsons,
    tool_names,
    toolset,
    tool_docstrings,
    subfield_tools_usage_experience,
):
    """处理单个问题"""
    problem = train_dataset[int(id)]["problem"]
    solution = train_dataset[int(id)]["solution"]
    answer = train_dataset[int(id)]["answer"]
    prompt = get_tool_prompt_template.format(problem, Field, subfield, tool_set_string)
    response = chat_api(model, prompt, 0, "", temperature=0.0)
    called_idx = get_useful_tools(response)
    called_idx = [
        idx
        for idx in called_idx
        if isinstance(idx, int) and idx < len(tool_jsons) and idx >= 0
    ]
    called_idx = list(set(called_idx))

    called_tool_string = ""
    for idx in called_idx:
        if tool_docstrings[idx]:
            called_tool_string += '"""\n' + tool_docstrings[idx] + "\n"
        if "experience_pool" in tool_jsons[idx]:
            tool_experience_pool = tool_jsons[idx]["experience_pool"]
            called_tool_string += "\nExperience list\n" + tool_experience_pool
        called_tool_string += '\n"""\n'
        called_tool_string += toolset[idx] + "\n\n"

    examples_string = ""
    for idx in called_idx:
        called_t = toolset[idx]
        if called_t in subfield_tools_usage_experience:
            for exp in subfield_tools_usage_experience[called_t]:
                if exp["is_correct"]:
                    examples_string += (
                        f"# Question: {exp['problem']}\n```python\n{exp['code']}\n---\n"
                    )
                    break

    prompt = tpot_prompt_template.format(
        tool_string=called_tool_string,
        examples_string=examples_string,
        question=problem,
        solution=solution,
    )
    response = chat_api(model, prompt, 0, "", temperature=0.0)
    code = extract_program(response)
    if "print" not in code:
        code += "\nprint(solution())"

    # 创建执行目录
    exec_dir = CODE_EXEC_DIR / str(id)
    exec_dir.mkdir(parents=True, exist_ok=True)

    report, result = execute_code(
        "\n".join([toolset[idx] for idx in called_idx]), code, str(exec_dir)
    )
    is_correct = False
    if report:
        is_correct = grade_answer(result, answer)

    actual_called_idx = []
    for idx in called_idx:
        for line in code.split("\n"):
            line = line.strip()
            if not line.startswith(("import", "from", "#")):
                if tool_names[idx] + "(" in line:
                    actual_called_idx.append(idx)
                    break

    return (
        id,
        is_correct,
        report,
        result,
        actual_called_idx,
        called_tool_string,
        code,
        problem,
        answer,
    )


def evaluate(
    Iter,
    subfield_ids,
    Field,
    subfield,
    current_tools,
    Q_subfield_set,
    subfield_tools_usage_experience,
):
    """评估工具集性能"""
    without_tool_ids = []
    TSC = 0
    TA = 0
    ids = subfield_ids[subfield]

    tool_jsons = current_tools[subfield]
    toolset = [tool_json["tool"] for tool_json in tool_jsons]
    tool_descriptions = [tool_json["description"] for tool_json in tool_jsons]
    tool_docstrings = [tool_json["docstring"] for tool_json in tool_jsons]
    tool_names = [tool_json["tool_name"] for tool_json in tool_jsons]

    tool_set_string = ""
    for idx, t in enumerate(toolset):
        tool_set_string += (
            f"No. {idx}:\n{tool_descriptions[idx]}\n```python\n{t}\n```\n"
        )

    model = "gpt-3.5-turbo"
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(
                process_problem,
                id,
                tool_set_string,
                model,
                tool_jsons,
                tool_names,
                toolset,
                tool_docstrings,
                subfield_tools_usage_experience,
            ): id
            for id in ids
        }

        for future in as_completed(futures):
            (
                id,
                is_correct,
                report,
                result,
                actual_called_idx,
                called_tool_string,
                code,
                problem,
                answer,
            ) = future.result()

            if actual_called_idx:
                for idx in actual_called_idx:
                    current_tools[subfield][idx]["Freq"] += 1
                    current_tools[subfield][idx]["TSR"] += 1 if is_correct else 0
                    tool = current_tools[subfield][idx]["tool"]
                    if tool in subfield_tools_usage_experience:
                        subfield_tools_usage_experience[tool].append(
                            {
                                "iter": Iter,
                                "id": id,
                                "is_correct": is_correct,
                                "problem": problem,
                                "answer": answer,
                                "report": report,
                                "result": result,
                                "code": code,
                            }
                        )
                    else:
                        subfield_tools_usage_experience[tool] = [
                            {
                                "iter": Iter,
                                "id": id,
                                "is_correct": is_correct,
                                "problem": problem,
                                "answer": answer,
                                "report": report,
                                "result": result,
                                "code": code,
                            }
                        ]
            else:
                without_tool_ids.append(id)
            if is_correct:
                TA += 1

            print(
                "id: ",
                id,
                "is_correct: ",
                is_correct,
                "report: ",
                report,
                "result: ",
                result,
                "used_tools: ",
                actual_called_idx,
            )

    TA = TA / len(ids)
    TSC = (len(ids) - len(without_tool_ids)) / len(ids)

    Q_values = []
    for tool_json in current_tools[subfield]:
        if tool_json["Freq"] == 0:
            Q_values.append(1)
        else:
            Q_values.append(1 - (tool_json["TSR"] / tool_json["Freq"]))
    if len(current_tools[subfield]) < 10:
        Q_values += [1] * (10 - len(current_tools[subfield]))
    Q_values = np.array(Q_values)
    Q_set = len(without_tool_ids) + 15 * (1 - TA) * len(ids)
    loss = calculate_loss(
        Q_values,
        Q_set,
        max(1, len(ids) / max(1, len(tool_jsons))),
        1.25,
        len(ids) / max(1, len(tool_jsons)),
        len(tool_jsons),
        10,
    )
    Q_subfield_set[subfield].append(
        {
            "iter": Iter,
            "Q_tool_loss": int(Q_values.sum(keepdims=False)),
            "TSC": TSC,
            "TA": TA,
            "n": len(current_tools[subfield]),
            "len_questions": len(ids),
            "len_without_tool_questions": len(without_tool_ids),
            "loss": loss,
            "without_tool_ids": without_tool_ids,
        }
    )

    # 保存结果
    output_dir = OPTIMIZE_TOOLSET_DIR / Field
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_json(current_tools, output_dir / f"subfield_tools_iter_{Iter}.json")
    dump_json(Q_subfield_set, output_dir / "Q_subfield_set.json")
    dump_json(
        subfield_tools_usage_experience,
        output_dir / "subfield_tools_usage_experience.json",
    )

    print(
        "Iter: ",
        Iter,
        "TSC: ",
        TSC,
        "TA: ",
        TA,
        "loss: ",
        loss,
        "without_tool_ids: ",
        len(without_tool_ids),
        "toolset size: ",
        len(current_tools[subfield]),
    )


def load_data(Field, subfield):
    """加载数据"""
    subfield_cluster_ids = read_json(
        KNOWLEDGE_DIR / Field / "subfield_cluster_ids.json"
    )
    train_dataset = load_jsonl(DATASET_DIR / "math/train.jsonl")
    parsed_knowledge = load_jsonl(KNOWLEDGE_DIR / "train_knowledge_parsed.jsonl")
    return subfield_cluster_ids, train_dataset, parsed_knowledge


def initialize_files(Field, subfield_ids):
    """初始化文件"""
    output_dir = OPTIMIZE_TOOLSET_DIR / Field
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (output_dir / "subfield_tools_usage_experience.json").exists():
        subfield_tools_usage_experience = {}
        dump_json(
            subfield_tools_usage_experience,
            output_dir / "subfield_tools_usage_experience.json",
        )

    if not (output_dir / "Q_subfield_set.json").exists():
        Q_subfield_set = {s: [] for s in subfield_ids}
        dump_json(Q_subfield_set, output_dir / "Q_subfield_set.json")

    if not (output_dir / "all_update_actions.json").exists():
        all_update_actions = {s: [] for s in subfield_ids}
        dump_json(all_update_actions, output_dir / "all_update_actions.json")


def delete_tools(
    Field,
    subfield,
    current_tools,
    Q_subfield_set,
    subfield_tools_usage_experience,
    wrong_info=None,
):

    current_TSC = Q_subfield_set[subfield][-1]["TSC"]
    current_TA = Q_subfield_set[subfield][-1]["TA"]
    current_without_tool_ids = Q_subfield_set[subfield][-1]["without_tool_ids"]
    current_n = len(current_tools[subfield])
    tool_jsons = current_tools[subfield]

    tools_stats = ""
    for idx, tool_json in enumerate(tool_jsons):
        if tool_json["Freq"] > 0:
            tools_stats += f"No. {idx}:\n{tool_json['description']}\nUsage Frequency: {tool_json['Freq']}, Tool Calling Success Rate: {(tool_json['TSR']/tool_json['Freq'])*100}%\nTool code:\n```python\n{tool_json['tool']}\n```\n\n"
        else:
            tools_stats += f"No. {idx}:\n{tool_json['description']}\nUsage Frequency: 0, Tool Calling Success Rate: no data \nTool code:\n```python\n{tool_json['tool']}\n```\n\n"
    if wrong_info:
        wrong_info_string = ""
        for action in wrong_info["actions"]["delete"]:
            wrong_info_string += (
                f"{action['description']}:\n\n```python\n{action['tool']}\n```\n\n"
            )
        wrong_info_string = wrong_info_string.strip()
        wrong_TSC = wrong_info["Q_subfield_set"]["TSC"]
        wrong_TA = wrong_info["Q_subfield_set"]["TA"]
        prompt = delete_tool_prompt_template_with_wrong_info.format(
            field=Field,
            subfield=subfield,
            n=current_n,
            TSC=current_TSC,
            TA=current_TA,
            tools_stats=tools_stats,
            wrong_info=wrong_info_string,
            wrong_TSC=wrong_TSC,
            wrong_TA=wrong_TA,
        )
    else:
        prompt = delete_tool_prompt_template.format(
            field=Field,
            subfield=subfield,
            n=current_n,
            TSC=current_TSC,
            TA=current_TA,
            tools_stats=tools_stats,
        )
    response = chat_api("gpt-3.5-turbo", prompt, 0, "", temperature=0.3)
    delete_idx = get_useful_tools(response)
    delete_idx = [idx for idx in delete_idx if idx < len(tool_jsons) and idx >= 0]
    actions = {"delete": [tool_jsons[idx] for idx in delete_idx]}
    new_tool_jsons = [
        tool_json for idx, tool_json in enumerate(tool_jsons) if idx not in delete_idx
    ]

    return new_tool_jsons, actions


def add_tools(
    Field,
    subfield,
    new_tool_jsons,
    Q_subfield_set,
    subfield_tools_usage_experience,
    train_dataset,
    wrong_info=None,
):
    # Todo: 根据TSC、TA、当前工具集的数量和工具，以及未调用工具的问题，使用LLM判断是否需要增加新的这个知识点的工具，以提高工具集的覆盖范围；若要增加，应该增加什么工具？
    current_TSC = Q_subfield_set[subfield][-1]["TSC"]
    current_TA = Q_subfield_set[subfield][-1]["TA"]
    current_without_tool_ids = Q_subfield_set[subfield][-1]["without_tool_ids"]
    current_n = len(new_tool_jsons)
    tool_jsons = new_tool_jsons

    tools_stats = ""
    for idx, tool_json in enumerate(tool_jsons):
        tools_stats += f"No. {idx}: {tool_json['description']}:\n```python\n{tool_json['tool']}\n```\n\n"

    # todo 将uncoverd problems 改成对应的task
    unsolved_problems = ""
    if len(current_without_tool_ids) > 30:
        current_without_tool_ids = random.sample(current_without_tool_ids, 30)
    for id in current_without_tool_ids:
        problem = train_dataset[int(id)]["problem"]
        keypoints = ";".join(get_kps(id, Field, subfield))
        unsolved_problems += f"Problem: {problem} \n Task: {keypoints}\n\n---\n"
    if wrong_info:
        wrong_info_string = ""
        for action in wrong_info["actions"]["add"]:
            wrong_info_string += (
                f"{action['description']}:\n\n```python\n{action['tool']}\n```\n\n"
            )
        wrong_info_string = wrong_info_string.strip()
        wrong_TSC = wrong_info["Q_subfield_set"]["TSC"]
        wrong_TA = wrong_info["Q_subfield_set"]["TA"]
        prompt = add_tool_prompt_template_with_wrong_info.format(
            field=Field,
            subfield=subfield,
            n=current_n,
            TSC=current_TSC,
            TA=current_TA,
            tools_stats=tools_stats,
            unsolved_problems=unsolved_problems,
            wrong_info=wrong_info_string,
            wrong_TSC=wrong_TSC,
            wrong_TA=wrong_TA,
        )
    else:
        prompt = add_tool_prompt_template.format(
            field=Field,
            subfield=subfield,
            n=current_n,
            TSC=current_TSC,
            TA=current_TA,
            tools_stats=tools_stats,
            unsolved_problems=unsolved_problems,
        )
    response = chat_api("gpt-3.5-turbo", prompt, 0, "", temperature=0.3)
    all_add_tools = extract_math_tools(extract_program(response))

    actions = {"add": []}
    for add_t in all_add_tools:
        if valid_tool(add_t):
            add_t_json = get_tool_json(add_t)
            actions["add"].append(add_t_json)
            tool_jsons.append(add_t_json)

    return tool_jsons, actions


def process_tool_modification(
    tool_json, Field, subfield, subfield_tools_usage_experience
):

    if tool_json["Freq"] > 1 and (tool_json["TSR"] / tool_json["Freq"]) <= 0.9:
        tool = tool_json["tool"]
        tool_experience_pool = tool_json.get("experience_pool", "None")
        right_tool_callings = ""
        wrong_tool_callings = ""

        if tool in subfield_tools_usage_experience:
            stue = subfield_tools_usage_experience[tool]
            wrong_stue = [exp for exp in stue if not exp["is_correct"]]
            if len(wrong_stue) > 10:
                # 截取最后10个
                wrong_stue = wrong_stue[-10:]
            for experience in wrong_stue:
                wrong_tool_callings += f"Problem: {experience['problem']}\nAnswer: {experience['answer']}\n```python\n{experience['code']}\n```\n------\nOutput:{experience['result']}\n\n"

        prompt = modify_tool_prompt_template.format(
            field=Field,
            subfield=subfield,
            Freq=tool_json["Freq"],
            TSR=(tool_json["TSR"] / tool_json["Freq"]) * 100,
            docstring=tool_json["docstring"],
            tool=tool,
            wrong_tool_callings=wrong_tool_callings,
            experience_pool=tool_experience_pool,
        )
        response = chat_api("gpt-3.5-turbo", prompt, 0, "", temperature=0.3)
        modified_tools = extract_math_tools(extract_program(response))
        experience_pool = extract_experience_pool(response)

        for t in modified_tools:
            modified_tool_json = get_tool_json(t)
            if experience_pool:
                modified_tool_json["experience_pool"] = experience_pool

            if valid_tool(t):
                Freq = max(1, tool_json["Freq"] / 2)
                TSR = max(1, tool_json["TSR"] / 2)
                modified_tool_json["Freq"] = Freq
                modified_tool_json["TSR"] = TSR
                return {
                    "action": "update",
                    "original": tool_json,
                    "modified": modified_tool_json,
                }
            break
        if experience_pool:
            tool_json["experience_pool"] = experience_pool
            return {
                "action": "update_experience",
                "original": tool_json,
                "modified": tool_json,
            }
    return {"action": "none", "original": tool_json}


def modify_tools(
    Field, subfield, new_tool_jsons, Q_subfield_set, subfield_tools_usage_experience
):
    # Todo: 根据单个工具的Freq、TSR、运行反馈信息，工具本身的质量、通用性等，判断是否需要对工具进行调整，以提高工具的成功率；若要调整，应该调整什么？；若不需要调整，说明错误不是工具本身导致的，则考虑是需要维护经验池，防止再次出现类似错误。
    actions = {"update": [], "update_experience": []}

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_tool_modification,
                tool_json,
                Field,
                subfield,
                subfield_tools_usage_experience,
            )
            for tool_json in new_tool_jsons
        ]

        for future in as_completed(futures):
            result = future.result()
            if result["action"] == "update":
                actions["update"].append((result["original"], result["modified"]))
                for idx, tool_json in enumerate(new_tool_jsons):
                    if tool_json["tool_name"] == result["original"]["tool_name"]:
                        new_tool_jsons[idx] = result["modified"]
                        break
            elif result["action"] == "update_experience":
                for idx, tool_json in enumerate(new_tool_jsons):
                    if tool_json["tool_name"] == result["original"]["tool_name"]:
                        new_tool_jsons[idx] = result["modified"]
                        break

    return new_tool_jsons, actions


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--Field", type=str, help="The field of study", default="Geometry"
    )
    parser.add_argument(
        "--subfield",
        type=str,
        help="The subfield within the field",
        default="Conic Sections",
    )
    args = parser.parse_args()

    Field = args.Field
    subfield = args.subfield

    subfield_cluster_ids, train_dataset, parsed_knowledge = load_data(Field, subfield)
    max_iter_num = 5

    subfield_ids = {}
    for s in subfield_cluster_ids:
        subfield_ids[s] = []

    initialize_files(Field, subfield_ids)

    rool_back_num = 0
    for Iter in range(max_iter_num + 1):
        if rool_back_num >= 2:
            break

        ids = []
        for cluster in subfield_cluster_ids[subfield]:
            if len(subfield_cluster_ids[subfield][cluster]) > 35:
                for id in subfield_cluster_ids[subfield][cluster]:
                    fs, sfs = get_fields_subfields(id)
                    if Field in fs and subfield in sfs and len(sfs) == 1:
                        ids.append(id)
            else:
                ids.extend(subfield_cluster_ids[subfield][cluster])
        if len(ids) > 50:
            ids = random.sample(ids, 50)
        subfield_ids[subfield] = ids

        print("Field", Field, "subfield", subfield, "ids: ", len(ids))

        output_dir = OPTIMIZE_TOOLSET_DIR / Field
        current_tools = read_json(output_dir / f"subfield_tools_iter_{Iter}.json")
        Q_subfield_set = read_json(output_dir / "Q_subfield_set.json")
        subfield_tools_usage_experience = read_json(
            output_dir / "subfield_tools_usage_experience.json"
        )
        all_update_actions = read_json(output_dir / "all_update_actions.json")

        if Iter == max_iter_num:
            if (
                len(Q_subfield_set[subfield]) >= Iter + 1
                and Q_subfield_set[subfield][-1]["iter"] >= Iter
            ):
                if "state" in Q_subfield_set[subfield][Iter]:
                    rool_back_num += 1
                continue
        else:
            if (
                len(Q_subfield_set[subfield]) >= Iter + 1
                and len(Q_subfield_set[subfield]) < max_iter_num + 1
            ) or (
                len(Q_subfield_set[subfield]) >= Iter + 1
                and len(Q_subfield_set[subfield]) >= max_iter_num + 1
            ):
                if "state" in Q_subfield_set[subfield][Iter]:
                    rool_back_num += 1
                continue

        if rool_back_num >= 2:
            break

        if (
            Iter == max_iter_num
            and len(Q_subfield_set[subfield])
            and Q_subfield_set[subfield][-1]["iter"] != max_iter_num
        ):
            evaluate(
                Iter,
                subfield_ids,
                Field,
                subfield,
                current_tools,
                Q_subfield_set,
                subfield_tools_usage_experience,
            )
        elif Iter < max_iter_num:
            if (len(Q_subfield_set[subfield]) == 0) or (
                len(Q_subfield_set[subfield])
                and Q_subfield_set[subfield][-1]["iter"] <= Iter
            ):
                evaluate(
                    Iter,
                    subfield_ids,
                    Field,
                    subfield,
                    current_tools,
                    Q_subfield_set,
                    subfield_tools_usage_experience,
                )

        current_tools = read_json(output_dir / f"subfield_tools_iter_{Iter}.json")
        Q_subfield_set = read_json(output_dir / "Q_subfield_set.json")
        subfield_tools_usage_experience = read_json(
            output_dir / "subfield_tools_usage_experience.json"
        )

        if (output_dir / f"subfield_tools_iter_{Iter+1}.json").exists():
            new_tools = read_json(output_dir / f"subfield_tools_iter_{Iter+1}.json")
        else:
            new_tools = {s: [] for s in current_tools}

        actions = {"Iter": Iter, "delete": [], "add": [], "update": []}
        state = ""
        wrong_info = {}

        if Iter > 0:
            last_loss = Q_subfield_set[subfield][-2]["loss"]
            current_loss = Q_subfield_set[subfield][-1]["loss"]
            if current_loss > last_loss:
                wrong_actions = all_update_actions[subfield][-1]
                wrong_q_subfield_set = Q_subfield_set[subfield][-1]
                wrong_tool_jsons = current_tools[subfield]
                wrong_info["actions"] = wrong_actions
                wrong_info["Q_subfield_set"] = wrong_q_subfield_set
                wrong_info["tool_jsons"] = wrong_tool_jsons
                state = "rollback"

                current_tools[subfield] = read_json(
                    output_dir / f"subfield_tools_iter_{Iter-1}.json"
                )[subfield]
                Q_subfield_set[subfield][-1] = copy.deepcopy(
                    Q_subfield_set[subfield][-2]
                )
                Q_subfield_set[subfield][-1]["state"] = "rollback"
                Q_subfield_set[subfield][-1]["iter"] = wrong_q_subfield_set["iter"]
                print("Iter: ", Iter, " rollback!", "loss: ", current_loss, last_loss)

        if state != "rollback":
            if Iter == max_iter_num:
                break
            rool_back_num = 0
            new_tool_jsons, delete_actions = delete_tools(
                Field,
                subfield,
                current_tools,
                Q_subfield_set,
                subfield_tools_usage_experience,
            )
            actions["delete"].extend(delete_actions["delete"])
            print("Iter: ", Iter, " delete:", len(delete_actions["delete"]))

            new_tool_jsons, modify_actions = modify_tools(
                Field,
                subfield,
                new_tool_jsons,
                Q_subfield_set,
                subfield_tools_usage_experience,
            )
            actions["update"].extend(modify_actions["update"])
            print("Iter: ", Iter, " update:", len(modify_actions["update"]))

        else:
            rool_back_num += 1
            if rool_back_num >= 2 or Iter == max_iter_num:
                dump_json(
                    current_tools, output_dir / f"subfield_tools_iter_{Iter}.json"
                )
                dump_json(Q_subfield_set, output_dir / "Q_subfield_set.json")
                break

            new_tool_jsons, delete_actions = delete_tools(
                Field,
                subfield,
                current_tools,
                Q_subfield_set,
                subfield_tools_usage_experience,
                wrong_info,
            )
            actions["delete"].extend(delete_actions["delete"])
            print("Iter: ", Iter, " delete:", len(delete_actions["delete"]))

            new_tool_jsons, modify_actions = modify_tools(
                Field,
                subfield,
                new_tool_jsons,
                Q_subfield_set,
                subfield_tools_usage_experience,
            )
            actions["update"].extend(modify_actions["update"])
            print("Iter: ", Iter, " update:", len(modify_actions["update"]))

        if state == "rollback":
            dump_json(current_tools, output_dir / f"subfield_tools_iter_{Iter}.json")
            dump_json(Q_subfield_set, output_dir / "Q_subfield_set.json")

        new_tools[subfield] = new_tool_jsons
        all_update_actions[subfield].append(actions)

        dump_json(new_tools, output_dir / f"subfield_tools_iter_{Iter+1}.json")
        dump_json(all_update_actions, output_dir / "all_update_actions.json")

    print("over!!!")


if __name__ == "__main__":
    main()
