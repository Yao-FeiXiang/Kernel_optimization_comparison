import subprocess
import sys
import os

def main():
    source_file = "SGEMM.cu"
    executable = "SGEMM.exe"

    if not os.path.exists(source_file):
        print(f"找不到源文件: {source_file}")
        sys.exit(1)

    print(f"开始编译 {source_file} ...")
    
    # 增加 -Xcompiler "/utf-8" 解决中文编码警告
    compile_cmd = ["nvcc", source_file, "-o", executable, "-Xcompiler", "/utf-8"]
    
    # 调用 nvcc 编译
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if compile_result.returncode != 0:
        print("编译失败！编译器输出如下：")
        print("--- STDOUT ---")
        print(compile_result.stdout)
        print("--- STDERR ---")
        print(compile_result.stderr)
        sys.exit(1)
        
    print("编译成功，开始运行...\n")
    print("-" * 40)
    
    # 运行命令：.\SGEMM.exe
    run_cmd = [f".\\{executable}"]
    
    # 执行并实时将输出流重定向到控制台
    try:
        subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n执行报错，退出码: {e.returncode}")
    except FileNotFoundError:
        print(f"\n找不到可执行文件: {executable}")

if __name__ == "__main__":
    main()