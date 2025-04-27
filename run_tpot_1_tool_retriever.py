from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
import json
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
TPOT_DIR = DATA_DIR / "tpot"
TOOLSET_DIR = DATA_DIR / "toolset"
RETRIEVER_OUTPUT_DIR = TPOT_DIR / "retriever_output"
CALLED_TOOLS_DIR = TPOT_DIR / "called_tools"

# 确保目录存在
for dir_path in [
    DATA_DIR,
    DATASET_DIR,
    KNOWLEDGE_DIR,
    TPOT_DIR,
    TOOLSET_DIR,
    RETRIEVER_OUTPUT_DIR,
    CALLED_TOOLS_DIR,
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


def get_useful_tools(item: Dict) -> Tuple[Dict, List[int]]:
    """
    获取有用的工具

    Args:
        item: 查询项

    Returns:
        Tuple[Dict, List[int]]: (查询项, 调用的工具列表)
    """
    id = item["id"]
    field = item["field"]
    subfield = item["subfield"]

    # 检查是否已存在结果
    output_file = RETRIEVER_OUTPUT_DIR / f"{id}_{field}_{subfield}.json"
    if output_file.exists():
        logging.info(f"已存在结果: {output_file}")
        history_json = read_json(output_file)
        return item, history_json["called_tools"]

    query = item["query"]
    toolset_size = item["toolset_size"]
    sys_message = ""
    called_tools = []

    try_num = 0
    while try_num < 1:
        try:
            response = chat_api(model, query, 0, sys_message, tempreture)
            pattern = r"<answer>(.*?)</answer>"
            matches = re.findall(pattern, response)

            for match in matches:
                ans = eval(match)
                for idx in ans:
                    idx = int(idx)
                    called_tools.append(idx)

            called_tools = [
                idx for idx in called_tools if idx < toolset_size and idx >= 0
            ]
            called_tools = list(set(called_tools))
            try_num += 1

            if len(called_tools) > 0:
                break

        except Exception as e:
            logging.error(f"处理ID {id} 时发生错误: {e}")
            logging.error(f"响应内容: {response}")
            try_num += 1

    logging.info(f"ID {id} 调用的工具: {called_tools}")
    return item, called_tools


def process_test_field(test_field: str) -> None:
    """
    处理测试领域

    Args:
        test_field: 测试领域名称
    """
    try:
        # 设置路径
        called_tool_path = CALLED_TOOLS_DIR / f"test_{test_field}_called_tools.json"
        final_toolset_path = TOOLSET_DIR / "final_toolset.json"

        # 读取测试ID
        test_ids = read_json(KNOWLEDGE_DIR / "test_ids_with_kps.json")[test_field]

        # 读取知识解析结果
        global test_knowledge_parsed
        test_knowledge_parsed = load_jsonl(
            KNOWLEDGE_DIR / "test_knowledge_parsed.jsonl"
        )

        # 读取数学领域配置
        global math_domains
        math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")

        # 生成查询
        all_querys = []
        for id in test_ids:
            kps = get_kps_by_id(id)
            for kp in kps:
                Field = kp["filed"]
                Subfield = kp["subfield"]
                tool_jsons = read_json(final_toolset_path)[Field][Subfield]

                if len(tool_jsons) == 0:
                    continue

                tools_string = ""
                for idx, tool_json in enumerate(tool_jsons):
                    tools_string += (
                        f"No. {idx}:\n"
                        f"\"{tool_json['tool_name']}\": {tool_json['description']}"
                        f"```python\n{tool_json['tool']}\n```\n\n"
                    )

                problem = test_dataset[int(id)]["problem"]
                Knowledge_Points = ";\n".join(kp["knowledge_point"])
                prompt = get_tool_prompt_template.format(
                    problem, Knowledge_Points, Field, Subfield, tools_string
                )

                all_querys.append(
                    {
                        "id": id,
                        "field": Field,
                        "subfield": Subfield,
                        "toolset_size": len(tool_jsons),
                        "query": prompt,
                    }
                )

        logging.info(f"生成的查询数量: {len(all_querys)}")

        # 处理查询
        all_called_tools = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(get_useful_tools, item) for item in all_querys]
            for future in futures:
                item, called_tools = future.result()

                id = item["id"]
                field = item["field"]
                subfield = item["subfield"]

                if len(called_tools) == 0:
                    continue

                if id not in all_called_tools:
                    all_called_tools[id] = [
                        {
                            "field": field,
                            "subfield": subfield,
                            "called_tools": called_tools,
                        }
                    ]
                else:
                    all_called_tools[id].append(
                        {
                            "field": field,
                            "subfield": subfield,
                            "called_tools": called_tools,
                        }
                    )

        # 保存结果
        logging.info(f"调用的工具数量: {len(all_called_tools)}")
        dump_json(all_called_tools, called_tool_path)

        # 分析结果
        logging.info(
            f"测试ID数量: {len(test_ids)}, 调用工具ID数量: {len(all_called_tools)}"
        )

        # 分析每个ID对应的工具数量
        id_tool_num = {}
        for key in all_called_tools:
            id = key
            for x in all_called_tools[key]:
                if id not in id_tool_num:
                    id_tool_num[id] = len(x["called_tools"])
                else:
                    id_tool_num[id] += len(x["called_tools"])

        # 计算平均值
        total = sum(id_tool_num.values())
        average = total / len(id_tool_num) if id_tool_num else 0
        logging.info(f"平均每个ID调用的工具数量: {average:.2f}")

    except Exception as e:
        logging.error(f"处理测试领域 {test_field} 时发生错误: {e}")
        raise


def main():
    """主函数"""
    try:
        # 定义测试类型
        all_type = [
            "Algebra",
            "Intermediate Algebra",
            "Prealgebra",
            "Geometry",
            "Counting & Probability",
            "Precalculus",
            "Number Theory",
        ]

        # 加载测试数据集
        global test_dataset
        test_dataset = load_jsonl(DATASET_DIR / "math/test.jsonl")

        # 设置模型参数
        global model, tempreture
        model = "gpt-3.5-turbo"
        tempreture = 0.0

        # 处理每个测试领域
        for test_field in tqdm(all_type, desc="处理测试领域"):
            process_test_field(test_field)

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
