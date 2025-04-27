from utils.api import *
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from FlagEmbedding import BGEM3FlagModel
from utils.grader import *
import json
import re
import os
import glob
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple, Optional

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
TPOT_DIR = DATA_DIR / "tpot"
TOOLSET_DIR = DATA_DIR / "toolset"
OUTPUT_DIR = TPOT_DIR / "output"
RESULT_DIR = TPOT_DIR / "result"
CODE_EXEC_DIR = PROJECT_ROOT / "code_exec"

# 确保目录存在
for dir_path in [
    DATA_DIR,
    DATASET_DIR,
    KNOWLEDGE_DIR,
    TPOT_DIR,
    TOOLSET_DIR,
    OUTPUT_DIR,
    RESULT_DIR,
    CODE_EXEC_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_kps_by_id(id: str) -> List[Dict]:
    """
    根据ID获取知识点

    Args:
        id: 数据ID

    Returns:
        List[Dict]: 知识点列表
    """
    kps = []
    for line in test_knowledge_parsed:
        if line["id"] == id:
            field = line["field"]
            subfield = line["subfield"]
            if field in math_domains and subfield in math_domains[field]:
                kps.append(
                    {
                        "filed": field,
                        "subfield": subfield,
                        "knowledge_point": line["keypoints"],
                    }
                )
    return kps


def get_tool_usage_example(Field: str, tool_code: str, _problem: str) -> List[str]:
    """
    获取工具使用示例

    Args:
        Field: 领域
        tool_code: 工具代码
        _problem: 问题

    Returns:
        List[str]: 示例列表
    """
    examples = []
    tool_usage_experience_path = (
        TOOLSET_DIR / "optimize" / Field / "subfield_tools_usage_experience.json"
    )
    tool_usage_experience = read_json(tool_usage_experience_path)

    if tool_code in tool_usage_experience:
        experience_list = [
            exp for exp in tool_usage_experience[tool_code] if exp["is_correct"]
        ]

        # 选择与problem相关的top_6例子，使用语义相似性
        sampled_experience_list = []
        if len(experience_list) < 6:
            sampled_experience_list = experience_list
        else:
            exp_questions = []
            for exp in experience_list:
                question = exp["problem"]
                if question not in exp_questions:
                    exp_questions.append(exp["problem"])
                    sampled_experience_list.append(exp)

            if len(sampled_experience_list) > 6:
                new_sampled_experience_list = []
                # 在exp_questions中选择与_problem语义相似的例子
                text1 = [_problem]
                text2 = exp_questions
                with torch.no_grad():
                    embeddings1 = bge.encode(
                        text1,
                        batch_size=12,
                        max_length=8192,
                    )["dense_vecs"]
                    embeddings2 = bge.encode(
                        text2,
                        batch_size=12,
                        max_length=8192,
                    )["dense_vecs"]
                    embeddings1 = torch.from_numpy(np.array(embeddings1)).to("cuda")
                    embeddings2 = torch.from_numpy(np.array(embeddings2)).to("cuda")
                    sim = embeddings1 @ embeddings2.T
                    k = 6
                    topk_values, selected_idxs = torch.topk(sim, k=k)
                    selected_idxs = selected_idxs.squeeze(0).tolist()
                    for idx in selected_idxs:
                        new_sampled_experience_list.append(sampled_experience_list[idx])

                    # 按照难度排序，难度从大到小
                    levels = [
                        train_dataset[int(exp["id"])]["level"]
                        for exp in new_sampled_experience_list
                    ]
                    sorted_indices = sorted(
                        range(len(levels)), key=lambda i: levels[i], reverse=True
                    )
                    new_sampled_experience_list = [
                        new_sampled_experience_list[i] for i in sorted_indices
                    ]
                    sampled_experience_list = new_sampled_experience_list
            else:
                more_need = 6 - len(sampled_experience_list)
                sampled_experience_list.extend(
                    random.sample(experience_list, more_need)
                )

        for exp in sampled_experience_list:
            question = exp["problem"]
            code = exp["code"]
            if "\nprint(solution())\n" in code:
                code = code.replace(
                    "\nprint(solution())\n",
                    "\n# Output the result\nprint(f'The result is: {solution()}\n",
                )
            examples.append(f"# Question: {question}\n\n```python\n{code}\n```")

    return examples


def get_function_name(code: str) -> Optional[str]:
    """
    获取函数名

    Args:
        code: 代码

    Returns:
        Optional[str]: 函数名
    """
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    if match:
        return match.group(1)
    return None


def process_item(
    item: Dict,
) -> Tuple[Dict, str, str, List[str], str, int, bool, str, str]:
    """
    处理单个项目

    Args:
        item: 项目数据

    Returns:
        Tuple[Dict, str, str, List[str], str, int, bool, str, str]: 处理结果
    """
    id = item["id"]
    sys_message = ""
    tools_string = item["tools_string"]
    output_path = OUTPUT_DIR / f"{id}.json"

    if output_path.exists():
        try:
            history_json = read_json(output_path)
            if len(history_json):
                logging.info(f"ID {id} 已存在结果")
                return (
                    item,
                    history_json["code"],
                    history_json["tools_string"],
                    history_json["real_called_tools"],
                    history_json["hint"],
                    history_json["first_report"],
                    history_json["report"],
                    history_json["result"],
                    history_json["api_state"],
                )
        except Exception as e:
            logging.error(f"读取历史结果时发生错误: {e}")
            pass

    if item["type"] == "pot":
        first_report = 4
        pot_prompt = pot_prompt_template.replace("===question===", item["problem"])
        n = 0
        while n <= 3:
            response = chat_api("", pot_prompt, 0, sys_message, tempreture)
            code = extract_program(response)
            if "print" not in code:
                code += "\nprint(f'The result is: {solution()}')"
            report, result = execute_code(
                tools_string, code, code_file=str(CODE_EXEC_DIR / id)
            )
            if report:
                first_report = n
                break
            n += 1
        logging.info(f"ID {id} 处理结果: {report}, {result}")

        hint = "None"
        if report:
            hint = result.strip()
            if len(hint) > 2000:
                hint = hint[:2000]
            cot_prompt = cot_prompt_template.replace("===qst===", item["problem"])
            cot_prompt = cot_prompt.replace("===hint===", hint)
        else:
            cot_prompt = cot_prompt_template.replace("===qst===", item["problem"])
            cot_prompt = cot_prompt.replace("===hint===", "None")

        response = chat_api(model, cot_prompt, 0, sys_message, 0.0)
        answer = extract_answer(response)
        if len(answer) == 0:
            report = False
            result = answer
        else:
            report = True
            result = answer

        api_state = "success"
        if response == "None":
            api_state = "fail"

        dump_json(
            {
                "id": id,
                "code": code,
                "type": item["type"],
                "tools_string": tools_string,
                "real_called_tools": [],
                "hint": hint,
                "report": report,
                "first_report": first_report,
                "result": result,
                "api_state": api_state,
            },
            output_path,
        )
        return (
            item,
            code,
            tools_string,
            [],
            hint,
            first_report,
            report,
            result,
            api_state,
        )

    tools_code = item["tools_code"]
    tpot_examples = item["tpot_examples"]
    problem = item["problem"]
    tpot_prompt = tpot_prompt_template.replace("===question===", problem)
    tpot_prompt = tpot_prompt.replace("===tool_string===", tools_string)
    tpot_prompt = tpot_prompt.replace("===examples===", "\n---\n".join(tpot_examples))

    n = 0
    first_report = 4
    while n <= 3:
        response = chat_api(model, tpot_prompt, 0, sys_message, tempreture)
        code = extract_program(response)
        if "print" not in code:
            code += "\nprint(f'The result is: {solution()}')"
        report, result = execute_code(
            "\n".join(tools_code), code, code_file=str(CODE_EXEC_DIR / id)
        )
        if report:
            first_report = n
            break
        n += 1
    logging.info(f"ID {id} 处理结果: {report}, {result}")

    hint = "None"
    real_called_tools = []
    # 判断是否使用了工具，若使用了工具则记录下来，计算工具使用频率
    for tc in tools_code:
        t_name = get_function_name(tc)
        if t_name:
            t_name = t_name.strip()
            if t_name + "(" in code:
                real_called_tools.append(tc)

    if report:
        hint = result.strip()
        if len(hint) > 2000:
            hint = hint[:2000]
        cot_prompt = cot_prompt_template.replace("===qst===", item["problem"])
        cot_prompt = cot_prompt.replace("===hint===", hint)
    else:
        cot_prompt = cot_prompt_template.replace("===qst===", item["problem"])
        cot_prompt = cot_prompt.replace("===hint===", "None")

    response = chat_api(model, cot_prompt, 0, sys_message, 0.0)
    answer = extract_answer(response)
    if len(answer) == 0:
        report = False
        result = answer
    else:
        report = True
        result = answer

    api_state = "success"
    if response == "None":
        api_state = "fail"

    dump_json(
        {
            "id": id,
            "code": code,
            "type": item["type"],
            "tools_string": tools_string,
            "real_called_tools": real_called_tools,
            "hint": hint,
            "report": report,
            "first_report": first_report,
            "result": result,
            "api_state": api_state,
        },
        output_path,
    )
    return (
        item,
        code,
        tools_string,
        real_called_tools,
        hint,
        first_report,
        report,
        result,
        api_state,
    )


def main():
    """主函数"""
    try:
        model = "gpt-3.5-turbo-0125"
        tempreture = 0.3
        bge = BGEM3FlagModel(
            r"BAAI/bge-m3",
            use_fp16=True,
            device="cuda:0",
        )

        test_field = sys.argv[1]
        test_dataset = load_jsonl(DATASET_DIR / "math/test.jsonl")
        train_dataset = load_jsonl(DATASET_DIR / "math/train.jsonl")

        result_path = (
            RESULT_DIR / f"test_tpot_{test_field}_all_results_tmp_{tempreture}.json"
        )
        if result_path.exists():
            logging.info(f"测试领域 {test_field} 的结果已存在")
            # 显式释放 GPU 内存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return

        test_ids_with_kps = read_json(KNOWLEDGE_DIR / "test_ids_with_kps.json")[
            test_field
        ]
        test_ids = [
            str(idx)
            for idx, line in enumerate(test_dataset)
            if line["type"] == test_field
        ]
        logging.info(
            f"测试ID数量: {len(test_ids)}, 带知识点的测试ID数量: {len(test_ids_with_kps)}"
        )

        # 加载输入信息
        global test_knowledge_parsed
        test_knowledge_parsed = load_jsonl(
            DATASET_DIR / "math/test_knowledge_parsed.jsonl"
        )
        global math_domains
        math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")
        all_call_tools = read_json(
            TPOT_DIR / "called_tools" / f"test_{test_field}_called_tools.json"
        )

        all_results = {}
        tpot_querys = []

        for id in test_ids:
            problem = test_dataset[int(id)]["problem"]
            kps = get_kps_by_id(id)
            if len(kps) == 0:
                kps = "None"

            if id not in all_call_tools:
                tpot_querys.append(
                    {
                        "id": id,
                        "type": "pot",
                        "tools_string": "",
                        "problem": problem,
                        "kps": kps,
                    }
                )
            else:
                tools_string = ""
                tools_code = []
                tpot_examples = []
                for json_y in all_call_tools[id]:
                    Field = json_y["field"]
                    Subfield = json_y["subfield"]
                    toolset = read_json(TOOLSET_DIR / "final_toolset.json")[Field][
                        Subfield
                    ]
                    called_tools = json_y["called_tools"]

                    for idx in called_tools:
                        if idx < 0 or idx > len(toolset) - 1:
                            logging.warning(
                                f"工具索引超出范围: idx={idx}, len(toolset)={len(toolset)}, id={id}"
                            )
                            continue

                        tool_json = toolset[idx]
                        tools_code.append(tool_json["tool"].strip())
                        if "experience_pool" in tool_json:
                            tools_string += f"\"\"\"\nTool Name: {tool_json['tool_name']}: \nField: {Field}\nSubfield: {Subfield}\nDocstring: {tool_json['docstring']}\nUsage Experience: {tool_json['experience_pool']}\n\"\"\"\n```python\n{tool_json['tool']}\n```\n\n"
                        else:
                            tools_string += f"\"\"\"\nTool Name: {tool_json['tool_name']}: \nField: {Field}\nSubfield: {Subfield}\nDocstring: {tool_json['docstring']}\n\"\"\"\n```python\n{tool_json['tool']}\n```\n\n"

                        tpot_examples_for_this_Code = get_tool_usage_example(
                            Field, tool_json["tool"], problem
                        )
                        if len(tpot_examples_for_this_Code) > 0:
                            tpot_examples.append(tpot_examples_for_this_Code)

                if len(tools_code) == 0:
                    tpot_querys.append(
                        {
                            "id": id,
                            "type": "pot",
                            "tools_string": "",
                            "problem": problem,
                            "kps": kps,
                        }
                    )
                else:
                    all_examples_nums = [len(x) for x in tpot_examples]
                    all_examples_nums = sum(all_examples_nums)
                    if all_examples_nums > 6:
                        # 从tpot_examples中选择6个例子，最好来自不同的工具
                        selected_examples = []
                        pointer = 0
                        while pointer <= 6 and len(selected_examples) < 6:
                            for idx, sublist_examples in enumerate(tpot_examples):
                                if len(sublist_examples) > pointer:
                                    selected_examples.append(sublist_examples[pointer])
                                    if len(selected_examples) >= 6:
                                        break
                            pointer += 1
                        tpot_examples = selected_examples
                    else:
                        # 添加默认例子
                        default_num = max(
                            len(tpot_examples_default), 6 - all_examples_nums
                        )
                        # 将二维列表转为一维
                        tpot_examples = [
                            example for sublist in tpot_examples for example in sublist
                        ]
                        tpot_examples.extend(tpot_examples_default[:default_num])

                    tpot_querys.append(
                        {
                            "id": id,
                            "type": "tpot",
                            "tools_string": tools_string,
                            "tools_code": tools_code,
                            "tpot_examples": tpot_examples,
                            "problem": problem,
                            "kps": kps,
                        }
                    )

        logging.info(f"查询数量: {len(tpot_querys)}")

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_item, item) for item in tpot_querys]
            for future in futures:
                (
                    item,
                    code,
                    tools_string,
                    real_called_tools,
                    hint,
                    first_report,
                    report,
                    result,
                    api_state,
                ) = future.result()
                id, query_type = item["id"], item["type"]
                all_results[id] = {
                    "query_type": query_type,
                    "code": code,
                    "tools_string": tools_string,
                    "real_called_tools": real_called_tools,
                    "hint": hint,
                    "first_report": first_report,
                    "report": report,
                    "result": result,
                    "api_state": api_state,
                }

        for id in all_results:
            line = all_results[id]
            answer = test_dataset[int(id)]["answer"][0]
            line["answer"] = answer
            line["is_correct"] = False
            if line["report"] and grade_answer(line["result"], answer):
                line["is_correct"] = True

        logging.info(f"结果数量: {len(all_results)}")
        dump_json(all_results, result_path)

        # 显式释放 GPU 内存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
