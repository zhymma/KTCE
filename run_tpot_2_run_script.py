import subprocess
from concurrent.futures import ThreadPoolExecutor
import torch
from pathlib import Path
import logging
from typing import List
import sys

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

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


def run_script(test_field: str) -> None:
    """
    运行测试脚本

    Args:
        test_field: 测试领域名称
    """
    try:
        # 构建脚本路径
        script_path = PROJECT_ROOT / "run_tpot_2_run.py"

        if not script_path.exists():
            raise FileNotFoundError(f"脚本文件不存在: {script_path}")

        # 运行脚本
        logging.info(f"开始处理测试领域: {test_field}")
        result = subprocess.run(
            [sys.executable, str(script_path), test_field],
            check=True,
            capture_output=True,
            text=True,
        )

        # 记录输出
        if result.stdout:
            logging.info(f"标准输出:\n{result.stdout}")
        if result.stderr:
            logging.error(f"错误输出:\n{result.stderr}")

        logging.info(f"完成处理测试领域: {test_field}")

    except subprocess.CalledProcessError as e:
        logging.error(f"脚本执行失败: {e}")
        logging.error(f"错误输出: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"处理测试领域 {test_field} 时发生错误: {e}")
        raise


def cleanup_gpu() -> None:
    """清理GPU内存"""
    try:
        logging.info("开始清理GPU内存")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.info("GPU内存清理完成")
    except Exception as e:
        logging.error(f"清理GPU内存时发生错误: {e}")
        raise


def main():
    """主函数"""
    try:
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有任务
            futures = [
                executor.submit(run_script, test_field) for test_field in ALL_TYPES
            ]

            # 等待所有任务完成
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"任务执行失败: {e}")
                    continue

        # 清理GPU内存
        cleanup_gpu()

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
