import os
import datetime
import subprocess
import platform

def get_base_filename(file_path):
    return  os.path.basename(file_path)

def get_file_dir(file_path):
    return  os.path.dirname(file_path)

def file_extension(file_path):
    file_name=os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    file_name_ext = os.path.splitext(file_name)[1]
    #file_name, file_extension = os.path.splitext(file_path)
    return file_name_no_ext, file_name_ext


def find_rvc_index_file(folder_path):
    for f in os.listdir(folder_path):
        if f.endswith('.index'):
            return f
    return None

def clean_not_latest_file(folder_path):

    # 获取文件夹中所有文件的路径和修改时间
    files = [(os.path.join(folder_path, f), os.path.getmtime(os.path.join(folder_path, f))) for f in os.listdir(folder_path)]

    # 根据修改时间从大到小排序
    files.sort(key=lambda x: x[1], reverse=True)

    # 获取最新文件路径
    if len(files)<1:
        print(f'no checkpoints in {folder_path}')
        sys.exit()

    # 删除不是最新的文件
    if len(files) > 2:
        for file_path, file_time in files[2:]:
            os.remove(file_path)
            print('已删除文件：', file_path)

    first=1 if len(files) > 1 else 0
    newest_file_path = files[first][0]
    # 获取最新文件的修改时间
    newest_file_time = datetime.datetime.fromtimestamp(files[first][1])

    print('最新文件路径：', newest_file_path)
    print('最新文件修改时间：', newest_file_time)

    return newest_file_path

def delete_directory(directory):
    if os.path.exists(directory):
      for filename in os.listdir(directory):
          file_path = os.path.join(directory, filename)
          if os.path.isfile(file_path):
              os.remove(file_path)
          elif os.path.isdir(file_path):
              delete_directory(file_path)
      os.rmdir(directory)

def execute_command(command,desc=None,work_dir="."):
    print(f"execute cmd: {command},desc={desc}")
    #return_code = subprocess.call(command, shell=platform.system() != 'Windows',cwd=work_dir)
    return_code = subprocess.call(command, shell=True,cwd=work_dir)
    #return_code = subprocess.call(["bash","-c",command],cwd=work_dir)
    if return_code != 0:
        raise Exception(f"错误code：{return_code}")