import json
from concurrent.futures import ProcessPoolExecutor
from utils.parser import *
from utils.utils import *
from utils.api import *
from utils.grader import *
from pathlib import Path
import logging
from typing import Dict, List, Any

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
TOOLSET_DIR = DATA_DIR / "toolset"
OPTIMIZE_TOOLSET_DIR = TOOLSET_DIR / "optimize"
INITIAL_TOOLSET_DIR = TOOLSET_DIR / "initial"

# 确保目录存在
for dir_path in [
    DATA_DIR,
    KNOWLEDGE_DIR,
    TOOLSET_DIR,
    OPTIMIZE_TOOLSET_DIR,
    INITIAL_TOOLSET_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)


def read_json(file_path: Path) -> Dict:
    """
    读取JSON文件

    Args:
        file_path: JSON文件路径

    Returns:
        Dict: JSON数据
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"读取JSON文件失败: {e}")
        raise


def dump_json(data: Dict, file_path: Path) -> None:
    """
    保存JSON数据到文件

    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存JSON文件失败: {e}")
        raise


def process_field(Field: str, subfields: List[str]) -> tuple:
    """
    处理单个领域的数据

    Args:
        Field: 领域名称
        subfields: 子领域列表

    Returns:
        tuple: (初始工具集, 最终工具集)
    """
    field_initial_toolset = {}
    field_final_toolset = {}

    try:
        # 读取Q_subfield_set
        q_subfield_set = read_json(OPTIMIZE_TOOLSET_DIR / Field / "Q_subfield_set.json")

        # 读取初始工具集
        initial_tools = read_json(
            OPTIMIZE_TOOLSET_DIR / Field / "subfield_tools_iter_0.json"
        )

        for subfield in subfields:
            try:
                # 获取最后一次迭代的编号
                last_iter = q_subfield_set[subfield][-1]["iter"]

                # 读取初始工具
                initiao_tools = initial_tools[subfield]
                field_initial_toolset[subfield] = initiao_tools

                # 读取最终工具
                tools = read_json(
                    OPTIMIZE_TOOLSET_DIR
                    / Field
                    / f"subfield_tools_iter_{last_iter}.json"
                )[subfield]

                # 过滤工具（如果需要）
                # tools = [tool for tool in tools if not (tool["Freq"] > 2 and tool["TSR"] == 0)]
                tools = [tool for tool in tools]

                field_final_toolset[subfield] = tools

                logging.info(
                    f"{Field} -> {subfield}: 初始工具数 {len(initiao_tools)}, 最终工具数 {len(tools)}"
                )

            except Exception as e:
                logging.error(f"处理子领域 {subfield} 时发生错误: {e}")
                continue

    except Exception as e:
        logging.error(f"处理领域 {Field} 时发生错误: {e}")
        raise

    return field_initial_toolset, field_final_toolset


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try:
        # 读取数学领域配置
        math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")

        # 初始化工具集
        initial_toolset = {}
        final_toolset = {}
        initial_tool_nums = 0

        # 使用进程池处理每个领域
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for Field, subfields in math_domains.items():
                future = executor.submit(process_field, Field, subfields)
                futures.append((Field, future))

            # 收集结果
            for Field, future in futures:
                try:
                    field_initial_toolset, field_final_toolset = future.result()
                    initial_toolset[Field] = field_initial_toolset
                    final_toolset[Field] = field_final_toolset

                    # 计算初始工具总数
                    for subfield in field_initial_toolset:
                        initial_tool_nums += len(field_initial_toolset[subfield])

                except Exception as e:
                    logging.error(f"处理领域 {Field} 的结果时发生错误: {e}")
                    continue

        # 保存结果
        dump_json(final_toolset, OPTIMIZE_TOOLSET_DIR / "final_toolset.json")
        dump_json(initial_toolset, INITIAL_TOOLSET_DIR / "initial_toolset.json")

        logging.info(f"初始工具总数: {initial_tool_nums}")

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
