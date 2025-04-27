import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 定义数据目录
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "dataset_knowledge_exact"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for dir_path in [DATA_DIR, KNOWLEDGE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def setup_logging(Field: str, subfield: str) -> None:
    """
    设置日志记录

    Args:
        Field: 领域名称
        subfield: 子领域名称
    """
    # 创建日志目录
    log_dir = LOGS_DIR / Field
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{subfield}_{timestamp}.log"

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, delay=False), logging.StreamHandler()],
        force=True,
    )

    # 确保日志消息立即写入文件
    for handler in logging.getLogger().handlers:
        handler.flush()

    logging.info(f"日志文件已创建: {log_file}")


def run_script_for_field(Field: str, subfields: list) -> None:
    """
    为指定领域运行优化脚本

    Args:
        Field: 领域名称
        subfields: 子领域列表
    """
    for subfield in subfields:
        try:
            setup_logging(Field, subfield)
            logging.info(f"开始处理 {Field} -> {subfield}")

            # 构建命令
            command = [
                "python",
                str(PROJECT_ROOT / "2_tool_creation_toolset_optimize.py"),
                "--Field",
                Field,
                "--subfield",
                subfield,
            ]

            # 执行命令
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,  # 检查命令是否成功执行
            )

            # 记录输出
            if result.stdout:
                logging.info(result.stdout)
            if result.stderr:
                logging.error(result.stderr)

            logging.info(f"完成处理 {Field} -> {subfield}")

        except subprocess.CalledProcessError as e:
            logging.error(f"命令执行失败: {e}")
            logging.error(f"错误输出: {e.stderr}")
        except Exception as e:
            logging.error(f"处理过程中发生错误: {e}")
        finally:
            # 确保日志被写入
            for handler in logging.getLogger().handlers:
                handler.flush()


def read_json(file_path: str) -> dict:
    """
    读取JSON文件

    Args:
        file_path: JSON文件路径

    Returns:
        dict: JSON数据
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"读取JSON文件失败: {e}")
        raise


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理数学领域和子领域的工具集优化")
    args = parser.parse_args()

    try:
        # 读取数学领域配置
        math_domains = read_json(KNOWLEDGE_DIR / "math_domains.json")

        # 使用进程池执行任务
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            total = len(math_domains)
            completed = 0

            # 提交任务
            for Field in math_domains:
                future = executor.submit(
                    run_script_for_field, Field, math_domains[Field]
                )
                futures.append(future)

            # 等待任务完成
            for future in as_completed(futures):
                completed += 1
                print(f"已完成 {completed}/{total} 个任务")
                try:
                    future.result()  # 获取任务结果，如果有异常会抛出
                except Exception as e:
                    logging.error(f"任务执行失败: {e}")

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
