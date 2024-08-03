import json
import random
from pathlib import Path

def sample_dataset(json_path: Path, output_path: Path, sample_ratio: float):
    """从数据集中抽取指定比例的数据
    
    Args:
        json_path (Path): 数据集 JSON 文件路径
        output_path (Path): 抽样后保存的 JSON 文件路径
        sample_ratio (float): 抽样比例，例如 0.001 表示 0.1%
    """
    print(f"Loading data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_count = len(data)
    sample_size = int(total_count * sample_ratio)
    print(f"Total data size: {total_count}, Sample size: {sample_size}")
    
    sampled_data = random.sample(data, sample_size)
    
    print(f"Saving sampled data to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)
    
    print(f"Sampling done, saved to {output_path}")

if __name__ == "__main__":
    DATA_ROOT = Path("/group_share/Ancient_Books/dataset")
    OUTPUT_PATH = "sampled_data.json"
    
    # Sample 0.1% from the final merged dataset
    FINAL_DATA_PATH = DATA_ROOT.joinpath("data.json")
    
    sample_dataset(
        json_path=FINAL_DATA_PATH,
        output_path=Path(OUTPUT_PATH),
        sample_ratio=0.001  # 0.1%
    )
