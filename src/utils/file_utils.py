# src/utils/file_utils.py
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the .yaml file.

    Returns:
        Dict[str, Any]: Parsed YAML content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gt(file_path: Union[str, Path]) -> List[str]:
    """
    Loads lines from a ground truth text file.

    Args:
        file_path (Union[str, Path]): Path to the .txt file.

    Returns:
        List[str]: List of lines from the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def save_txt(lines: List[str], output_path: Union[str, Path]) -> None:
    """
    Saves a list of strings to a text file. Ensures parent directories exist.

    Args:
        lines (List[str]): Content to write.
        output_path (Union[str, Path]): Destination file path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" if not line.endswith("\n") else line 
                     for line in lines)