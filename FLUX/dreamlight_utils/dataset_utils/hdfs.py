import os
import subprocess
import shutil
from typing import List

def listdir(path: str) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if path.startswith('hdfs://'):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files

def mkdir(path: str):
    """
    Create directory. Support either hdfs or local path.
    Create all parent directory if not present. No-op if directory already present.
    """
    if path.startswith('hdfs://'):
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)

def copy(src: str, tgt: str):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    src_hdfs = src.startswith("hdfs://")
    tgt_hdfs = tgt.startswith("hdfs://")

    if src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-cp", "-f", src, tgt])
    elif src_hdfs and not tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-copyToLocal", "-f", src, tgt])
    elif not src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-copyFromLocal", "-f", src, tgt])
    else:
        shutil.copy(src, tgt)

def exists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("hdfs dfs -test -e {}".format(file_path)) == 0
    return os.path.exists(file_path)
