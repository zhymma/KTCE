from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
import numpy as np

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
TPOT_DIR = DATA_DIR / "tpot"
TOOLSET_DIR = DATA_DIR / "toolset"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# 定义测试类型
ALL_TYPES = [
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Geometry",
    "Counting & Probability",
    "Precalculus",
    "Number Theory",
]


def load_toolset() -> Dict[str, Dict[str, List[Dict]]]:
    """
    加载工具集

    Returns:
        Dict[str, Dict[str, List[Dict]]]: 工具集数据
    """
    return read_json(TOOLSET_DIR / "final_toolset.json")


def initialize_tool_counts(toolset: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, int]:
    """
    初始化工具使用计数

    Args:
        toolset: 工具集数据

    Returns:
        Dict[str, int]: 工具使用计数
    """
    count_all_tool = {}
    for field in toolset:
        for subfield in toolset[field]:
            for tool in toolset[field][subfield]:
                count_all_tool[tool["tool"]] = 0
    return count_all_tool


def process_results(
    results: Dict[str, Dict], test_dataset: List[Dict], count_all_tool: Dict[str, int]
) -> Tuple[List[str], int, int, List[int], List[int]]:
    """
    处理结果数据

    Args:
        results: 结果数据
        test_dataset: 测试数据集
        count_all_tool: 工具使用计数

    Returns:
        Tuple[List[str], int, int, List[int], List[int]]: 处理结果
    """
    processed_id = []
    correct_count = 0
    pass_count = 0
    pass_correct_count = 0
    list_first_correct_count = []
    list_use_tool_nums = []

    for id in results:
        try:
            answer = test_dataset[int(id)]["answer"][0]
            processed_id.append(id)
        except Exception as e:
            logging.error(f"处理ID {id} 时发生错误: {e}")
            continue

    for id in results:
        if id not in processed_id:
            continue

        line = results[id]
        if line["is_correct"]:
            correct_count += 1
            if line["first_report"]:
                list_first_correct_count.append(line["first_report"])

        if line["report"] is True:
            pass_count += 1
            if line["is_correct"]:
                pass_correct_count += 1

        if "real_called_tools" in line and len(line["real_called_tools"]) > 0:
            list_use_tool_nums.append(len(line["real_called_tools"]))
            for t in line["real_called_tools"]:
                count_all_tool[t] += 1

    return (
        processed_id,
        correct_count,
        pass_correct_count,
        list_first_correct_count,
        list_use_tool_nums,
    )


def calculate_statistics(
    total_count: int,
    correct_count: int,
    pass_count: int,
    pass_correct_count: int,
    list_first_correct_count: List[int],
    list_use_tool_nums: List[int],
) -> Dict[str, float]:
    """
    计算统计数据

    Args:
        total_count: 总数
        correct_count: 正确数
        pass_count: 通过数
        pass_correct_count: 通过正确数
        list_first_correct_count: 首次正确列表
        list_use_tool_nums: 使用工具数列表

    Returns:
        Dict[str, float]: 统计数据
    """
    return {
        "correct_percentage": (correct_count / total_count) * 100,
        "first_correct_mean": np.array(list_first_correct_count).mean(),
        "use_tool_percentage": (len(list_use_tool_nums) / total_count) * 100,
        "use_tool_means": np.array(list_use_tool_nums).mean(),
        "pass_percentage": (pass_count / total_count) * 100,
        "pass_correct_percentage": (
            (pass_correct_count / pass_count) * 100 if pass_count > 0 else 0
        ),
    }


def print_statistics(type_name: str, stats: Dict[str, float], counts: Dict[str, int]):
    """
    打印统计数据

    Args:
        type_name: 类型名称
        stats: 统计数据
        counts: 计数数据
    """
    logging.info(f"Type: {type_name}")
    logging.info(f"Total count: {counts['total']}")
    logging.info(
        f"Total correct count: {counts['correct']} ({stats['correct_percentage']:.2f}%)"
    )
    logging.info(f"First correct mean: {stats['first_correct_mean']}")
    logging.info(
        f"Use tool count: {counts['use_tool']} ({stats['use_tool_percentage']:.2f}%)"
    )
    logging.info(f"Use tool mean: {stats['use_tool_means']}")
    logging.info(
        f"Total pass count: {counts['pass']} ({stats['pass_percentage']:.2f}%)"
    )
    logging.info(
        f"Total pass correct count: {counts['pass_correct']} ({stats['pass_correct_percentage']:.2f}%)"
    )
    logging.info("-" * 50)


def main():
    """主函数"""
    try:
        # 加载工具集
        final_toolset = load_toolset()
        count_all_tool = initialize_tool_counts(final_toolset)

        # 初始化全局统计
        all_count = 0
        all_correct_count = 0
        all_use_tool_count = 0

        # 加载测试数据集
        test_dataset = load_jsonl(DATASET_DIR / "math/test.jsonl")

        # 处理每个类型
        for type_name in ALL_TYPES:
            tempreture = 0.3
            model = "gpt-3.5-turbo-0125"

            # 加载结果
            results = read_json(
                TPOT_DIR
                / "result"
                / f"test_tpot_{type_name}_all_results_tmp_{tempreture}.json"
            )

            # 处理结果
            (
                processed_id,
                correct_count,
                pass_correct_count,
                list_first_correct_count,
                list_use_tool_nums,
            ) = process_results(results, test_dataset, count_all_tool)

            # 计算统计数据
            total_count = len(processed_id)
            pass_count = sum(
                1
                for id in results
                if id in processed_id and results[id]["report"] is True
            )

            stats = calculate_statistics(
                total_count,
                correct_count,
                pass_count,
                pass_correct_count,
                list_first_correct_count,
                list_use_tool_nums,
            )

            counts = {
                "total": total_count,
                "correct": correct_count,
                "pass": pass_count,
                "pass_correct": pass_correct_count,
                "use_tool": len(list_use_tool_nums),
            }

            # 打印统计信息
            print_statistics(type_name, stats, counts)

            # 更新全局统计
            all_count += total_count
            all_correct_count += correct_count
            all_use_tool_count += len(list_use_tool_nums)

        # 计算工具使用频率
        all_num = sum(count_all_tool.values())
        logging.info(
            f"Tool use Freq Mean: {all_num} {len(count_all_tool)} {all_num / len(count_all_tool)}"
        )

        # 打印全局统计
        logging.info(f"Total count: {all_count}")
        logging.info(
            f"Total correct count: {all_correct_count} ({(all_correct_count / all_count) * 100:.2f}%)"
        )
        logging.info(
            f"Use tool count: {all_use_tool_count} ({(all_use_tool_count / all_count) * 100:.2f}%)"
        )

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
