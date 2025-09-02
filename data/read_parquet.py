import pandas as pd
import os

def read_parquet_sample(file_path, num_rows=5):
    """
    读取parquet文件的前几行内容，显示数据格式
    
    Args:
        file_path (str): parquet文件路径
        num_rows (int): 要显示的行数，默认为5行
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        print(f"文件路径: {file_path}")
        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n" + "="*50)
        print("数据类型:")
        print(df.dtypes)
        print("\n" + "="*50)
        print(f"前{num_rows}行数据:")
        print(df.head(num_rows))
        print("\n" + "="*50)
        
        # 显示每列的详细信息
        print("列信息详情:")
        for col in df.columns:
            print(f"列名: {col}")
            print(f"  数据类型: {df[col].dtype}")
            print(f"  非空值数量: {df[col].notna().sum()}")
            
            # 安全地计算唯一值数量
            try:
                unique_count = df[col].nunique()
                print(f"  唯一值数量: {unique_count}")
            except Exception as e:
                print(f"  唯一值数量: 无法计算 ({e})")
            
            if df[col].dtype == 'object':
                # 对于文本类型，显示前几个样本值
                try:
                    # 获取非空值
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # 显示第一个值的类型和内容（截断显示）
                        first_value = non_null_values.iloc[0]
                        print(f"  第一个值的类型: {type(first_value)}")
                        
                        # 如果是字符串，显示前200个字符
                        if isinstance(first_value, str):
                            preview = first_value[:200] + "..." if len(first_value) > 200 else first_value
                            print(f"  第一个值预览: {preview}")
                        else:
                            # 如果是其他类型（如列表、字典），显示其结构
                            print(f"  第一个值内容: {str(first_value)[:200]}...")
                            
                except Exception as e:
                    print(f"  样本值: 无法显示 ({e})")
            print()
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 指定要读取的parquet文件路径
    parquet_file = "/data/RLAMG/data/valid.parquet"
    # parquet_file = "/data1/dataset/Openr1-Math-46k-8192/Openr1-Math-46k-Avg16.parquet"
    
    # 读取并显示前5行内容
    read_parquet_sample(parquet_file, num_rows=10)
