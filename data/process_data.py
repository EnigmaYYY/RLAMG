import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
SYSTEM_PROMPT = "You are a helpful assistant. Please reason step by step to solve the problem and put the final answer within the <answer> </answer> tags."

def convert_to_json_serializable(obj):
    """
    将pandas/numpy对象转换为JSON可序列化的格式
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def read_json_file(file_path):
    # 读取json文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except:
                print(line)
                1/0
    return data

def read_praquet_file(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    # 将每一行转换为字典，确保JSON可序列化
    result = [row.to_dict() for _, row in df.iterrows()]
    return result

def save_json(file_path,data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f'Save {file_path} is ok!')

def save_jsonl(file_path,data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")

def save_parquet(file_path, data):

    if isinstance(data, list):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame or a list of lists")
    pq.write_table(pa.Table.from_pandas(data), file_path)
    print(f'Save {file_path} is ok!')

def convert_parquet_to_json(parquet_file_path):
    """
    将parquet文件转换为json文件
    
    Args:
        parquet_file_path (str): parquet文件的路径
        
    Returns:
        str: 生成的json文件路径
    """
    try:
        # 检查parquet文件是否存在
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"Parquet文件不存在: {parquet_file_path}")
        
        # 使用现有的函数读取parquet文件
        print(f"正在读取parquet文件: {parquet_file_path}")
        data = read_praquet_file(parquet_file_path)
        
        # 转换为JSON可序列化的格式
        print("正在转换数据格式...")
        serializable_data = convert_to_json_serializable(data)
        
        # 生成json文件路径（同目录下，相同文件名，不同扩展名）
        base_name = os.path.splitext(parquet_file_path)[0]
        json_file_path = base_name + '.json'
        
        # 使用现有的函数保存为json文件
        save_json(json_file_path, serializable_data)
        
        print(f"成功转换: {parquet_file_path} -> {json_file_path}")
        print(f"共转换 {len(serializable_data)} 条数据")
        
        return json_file_path
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        raise e

def merge_json_files(directory_path=None):
    """合并多个JSON文件的数据"""
    merged_data = []
    
    # 如果提供了目录路径，读取该目录下所有JSON文件
    if directory_path and os.path.isdir(directory_path):
        json_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                     if f.endswith('.json')]
        for file_path in json_files:
            try:
                data = read_json_file(file_path)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
                print(f"成功读取文件: {file_path}, 包含 {len(data) if isinstance(data, list) else 1} 条数据")
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {str(e)}")
    
    print(f"合并完成，共 {len(merged_data)} 条数据")
    return merged_data

def extract_model_responses(models_list):
    """从所有模型中提取响应数据，按问题分组"""
    problem_responses = {}  # 以问题文本为key，存储来自不同模型的响应
    
    for model_name in models_list:
        print(f"正在提取模型 {model_name} 的响应...")
        model_data = merge_json_files(f"/home/yxy/RLAMG/outputs/data/openr1-math-46k-8192/{model_name}")
        
        if not model_data:
            print(f"模型 {model_name} 没有数据，跳过")
            continue
        
        extracted_count = 0
        for item in model_data:
            # 检查是否有正确回答
            if not item.get('correct_responses') or not item['correct_responses']:
                continue
            
            problem = item['question']['problem']
            
            # 只取第一个正确回答
            first_correct_response = item['correct_responses'][0]
            
            # 判断response格式
            if not first_correct_response.startswith('<think>') and '</think>' in first_correct_response:
                first_correct_response = '<think>\n' + first_correct_response
                # print(f'来自模型{model_name}, 响应中未使用<think>开头')
            if '<answer>' in first_correct_response or '</answer>' in first_correct_response:
                first_correct_response=first_correct_response.replace('<answer>', '')
                first_correct_response=first_correct_response.replace('</answer>', '')
                # print(f'来自模型{model_name}, 响应中有<answer>')
            
            # 重写数据格式   
            boxed_answer = item['boxed_answer']
            first_correct_response = first_correct_response + '\n' + '<answer>' + boxed_answer + '</answer>'
            
            # 计算质量分数：0.8（答案正确分）+ 响应长度分（0-0.2，长度越短分数越高）
            response_len = len(first_correct_response)
            # 响应长度分：设定一个基准长度，超过则分数递减
            max_length = 8192  # 设定最大期望长度
            # length_score = max(0, 0.2 * (1 - min(response_len / max_length, 1)))
            length_score = max(0, 0.2 * min(1, 0.1*(max_length / response_len)))
            quality_score = 0.8 + length_score
            
            # 获取ground truth
            ground_truth = item['question']['final_answer']
            
            # 如果这个问题还没有记录过，创建新条目
            if problem not in problem_responses:
                problem_responses[problem] = {
                    'problem': problem,
                    'ground_truth': ground_truth,
                    'responses': [],
                }
            
            # 添加这个模型的响应
            problem_responses[problem]['responses'].append({
                "content": first_correct_response,
                "role": "assistant",
                "source": f"{model_name}",
                "quality_score": round(quality_score, 3)
            })
            
            extracted_count += 1
        
        print(f"模型 {model_name} 提取完成，有效响应: {extracted_count} 条")
    
    return problem_responses

def process_combined_data(problem_responses):
    """将按问题分组的响应数据转换为最终格式"""
    processed_data = []
    
    # 系统提示词
    # system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    # system_prompt = "Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} <\/think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"<\/think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions."
    
    for idx, (problem, data) in enumerate(problem_responses.items()):
        # 构建prompt
        prompt = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"content": f"{problem}", "role": "user"}
        ]
        
        # 构建最终数据
        final_item = {
            "data_source": "openr1-math-46k",
            "prompt": prompt,
            "target_lst": data['responses'],
            "ability": "math",
            "reward_model": {
                "ground_truth": data['ground_truth'],
                "style": "rule"
            },
            "extra_info": {
                "index": -1,
                "split": "default",
                "num_teacher_models": len(data['responses'])
            }
        }
        
        processed_data.append(final_item)
    
    return processed_data

def get_train_data(models_list=None):
    """
    获取训练数据
    Args:
        models_list: 模型列表，如果为None则使用所有可用模型
        output_dir: 输出文件路径
    """
    if models_list is None:
        # 获取所有可用模型
        base_path = "/home/yxy/RLAMG/outputs/data/openr1-math-46k-8192"
        models_list = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"开始处理模型: {models_list}")
    
    # 第一步：从所有模型中提取响应数据，按问题分组
    problem_responses = extract_model_responses(models_list)
    print(f"提取完成，共找到 {len(problem_responses)} 个不同的问题")
    
    # 第二步：将分组数据转换为最终格式
    all_processed_data = process_combined_data(problem_responses)
    print(f"数据处理完成，最终有效数据: {len(all_processed_data)} 条")
    
    # 打印统计信息
    multi_model_data = [item for item in all_processed_data if item['extra_info']['num_teacher_models'] > 1]
    print(f"其中包含多个教师模型响应的问题: {len(multi_model_data)} 条")
    
    return all_processed_data, multi_model_data

def get_common_prompt_data(data1, data2):
    """
    提取两个parquet文件中prompt相同的数据
    
    Args:
        data1_path:文件1路径
        data2_path: 文件2路径
        output_path: 输出文件路径
    
    Returns:
        list: 过滤后的数据列表
    """
    
    print(f"data1数据量: {len(data1)}")
    print(f"data2数据量: {len(data2)}")
    print(data2[0]['prompt'][1])
    
    # 提取multi_teacher中的所有prompt作为过滤条件
    # 注意：prompt是一个列表，需要转换为字符串进行比较
    data2_prompts = []
    for item in data2:
        if 'prompt' in item:
            # 将prompt列表转换为字符串用于比较
            # prompt_str = json.dumps(item['prompt'][1], ensure_ascii=False, sort_keys=True)
            problem = item['prompt'][1]['content']
            data2_prompts.append(problem)
    
    print(f"从data2中提取到 {len(data2_prompts)} 个唯一prompt")
    
    # 从data1数据中筛选出prompt匹配的数据
    filtered_data = []
    matched_count = 0
    
    for item in data1:
        if 'prompt' in item:
            problem = item['prompt'][1]['content']
            
            if problem in data2_prompts:
                processed_item = process_openr1_data(item)
                filtered_data.append(processed_item)
                # filtered_data.append(item)
                matched_count += 1
    
    print(f"成功匹配到 {matched_count} 条数据")
    
    # # 保存过滤后的数据
    # if filtered_data:
    #     save_parquet(output_path, filtered_data)
    #     print(f"过滤后的数据已保存到: {output_path}")
    # else:
    #     print("没有找到匹配的数据")
    
    return filtered_data

def process_openr1_data(data):
    problem = data['prompt'][1]['content']
    # system_prompt = data['prompt'][0]['content']
    # system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final solution and answer. The reasoning process and answer are enclosed within <think> </think>, <solution> </solution> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <solution> solution here </solution> <answer> answer here </answer>."
    response = data['target'][0]['content']
    
    from match_answer import extract_boxed_answer
    boxed_answer = extract_boxed_answer(response)
    
    # 判断response格式
    if not response.startswith('<think>') and '</think>' in response:
        response = '<think>\n' + response
        # print(f'来自模型{model_name}, 响应中未使用<think>开头')
            
    # 重写数据格式
    response = f'{response}\n<answer>{boxed_answer}</answer>'
    
    # 构建prompt
    prompt = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": problem, "role": "user"}
    ]
    
    target = [{"content": response, "role": "assistant"}]
        
    # 构建最终数据
    processed_data = {
        "data_source": data['data_source'],
        "prompt": prompt,
        "target": target,
        "ability": data['ability'],
        "reward_model": data['reward_model'],
        "extra_info": data['extra_info']
    }
    return processed_data

def add_openr1_data(openr1_data_path, data_path, output_path):
    """
    将 openr1 数据中的 target 内的 r1 模型响应添加至 data 的 target_lst 中
    """
    print("开始读取parquet文件...")
    
    # 读取两个parquet文件
    openr1_data = read_praquet_file(openr1_data_path)
    data = read_praquet_file(data_path)
    
    print(f"openr1_data数据量: {len(openr1_data)}")
    print(f"data数据量: {len(data)}")
    
    # 构建 openr1 prompt -> response 的映射
    openr1_map = {}
    for row in openr1_data:
        problem = row['prompt'][1]['content']
        openr1_map[problem] = row['target'][0]['content']
    
    processed_data = []
    for item in data:
        problem = item['prompt'][1]['content']
        
        if problem not in openr1_map:
            continue
        
        response = openr1_map[problem]
        # 计算质量分数
        response_len = len(response) if response else 0
        max_length = 8192
        length_score = max(0, 0.2 * min(1, 0.1 * (max_length / max(1, response_len))))
        quality_score = 0.8 + length_score
        
        # 安全获取原有 target_lst（可能字段名不同）
        orig_target_lst = item['target_lst']
        target_lst = list(orig_target_lst)
        
        # 新的 r1 目标（字典形式）
        r1_target = {
            "content": response,
            "role": "assistant",
            "source": "DeepSeek-R1",
            "quality_score": round(quality_score, 3)
        }
        target_lst.append(r1_target)
        
        extra_info = {
            "index": -1,
            "split": "default",
            "num_teacher_models": len(target_lst)
        }
        
        processed_item = {
            "data_source": item.get('data_source', 'openr1'),
            "prompt": item.get('prompt'), 
            "target_lst": target_lst,
            "ability": item.get('ability'),
            "reward_model": item.get('reward_model', {}),
            "extra_info": extra_info
        }
        processed_data.append(processed_item)
    
    print(f"成功获取到 {len(processed_data)} 条数据")
    
    # 保存过滤后的数据
    if processed_data:
        save_parquet(output_path, processed_data)
        print(f"过滤后的数据已保存到: {output_path}")
    
    return processed_data

def process_valid_data(data, output_path):
    # print("开始读取parquet文件...")
    # data = read_praquet_file(data_path)
    new_data = []
    for item in data:
        problem = item['prompt'][1]['content']
        
        # 构建prompt
        prompt = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"content": problem, "role": "user"}
        ]
            
        # 构建最终数据
        processed_item = {
            "data_source": item['data_source'],
            "prompt": prompt,
            "ability": item['ability'],
            "reward_model": item['reward_model'],
            "extra_info": item['extra_info']
        }
        new_data.append(processed_item)
    save_parquet(output_path, new_data)
    # save_json("/data/RLAMG/data/valid_all.json", new_data)
    print(f"处理后的数据已保存到: {output_path}")
    return new_data

def replace_system_prompt(data, output_path):
    for item in data:
        item['prompt'][0]['content'] = SYSTEM_PROMPT

    save_parquet(output_path, data)
    print(f"处理后的数据已保存到: {output_path}")

def filter_data_by_models(data_path, model_list, output_path=None):
    """
    从数据集中筛选出包含指定模型响应的数据
    
    Args:
        data_path (str): 输入数据文件路径
        model_list (list): 指定的模型名称列表
        output_path (str, optional): 输出文件路径，如果为None则不保存文件
    
    Returns:
        list: 筛选后的数据列表
    """
    print(f"开始读取数据文件: {data_path}")
    data = read_praquet_file(data_path)
    print(f"原始数据量: {len(data)}")
    print(f"指定的模型列表: {model_list}")
    
    filtered_data = []
    valid_count = 0
    
    for item in data:
        # 获取当前问题的所有响应来源
        target_lst = item.get('target_lst', [])
        response_sources = [response.get('source', '') for response in target_lst]
        
        # 检查是否包含所有指定的模型
        has_all_models = all(model in response_sources for model in model_list)
        
        if has_all_models:
            # 只保留指定模型的响应
            filtered_responses = []
            for response in target_lst:
                if response.get('source', '') in model_list:
                    filtered_responses.append(response)
            
            # 更新数据项
            new_item = item.copy()
            new_item['target_lst'] = filtered_responses
            new_item['extra_info'] = new_item.get('extra_info', {}).copy()
            new_item['extra_info']['num_teacher_models'] = len(filtered_responses)
            
            filtered_data.append(new_item)
            valid_count += 1
    
    print(f"筛选完成:")
    print(f"  - 包含所有指定模型响应的问题数: {valid_count}")
    print(f"  - 筛选后数据量: {len(filtered_data)}")
    print(f"  - 保留率: {len(filtered_data)/len(data)*100:.2f}%")
    
    # 打印每个模型的响应数统计
    model_counts = {model: 0 for model in model_list}
    for item in filtered_data:
        for response in item['target_lst']:
            source = response.get('source', '')
            if source in model_counts:
                model_counts[source] += 1
    
    print("各模型响应数统计:")
    for model, count in model_counts.items():
        print(f"  - {model}: {count}")
    
    # 保存文件（如果指定了输出路径）
    if output_path:
        save_parquet(output_path, filtered_data)
        print(f"筛选后的数据已保存到: {output_path}")
        
        # 同时保存json格式
        json_output_path = output_path.replace('.parquet', '.json')
        convert_parquet_to_json(output_path)
    
    return filtered_data

def get_sft_data(data):
    sft_data = []
    for item in data:
        question = item['prompt'][1]['content']
        responses = [target['content'] for target in item['target_lst']]
        for response in responses:
            sft_data.append({
                "instruction": question,
                "input": "",
                "output": response,
                "system": SYSTEM_PROMPT
            })
    return sft_data

if __name__ == "__main__":
    # get train_data
    # longcot_models = ['Qwen3-8B_thinking', 'DeepSeek-R1-Distill-Qwen-7B', 'OpenR1-Qwen-7B', 'AceReason-Nemotron-1.1-7B']
    # shortcot_models = ['Qwen3-8B', 'Qwen2.5-Math-7B-Instruct', 'Qwen2.5-Math-7B-Oat-Zero', 'SynLogic-7B']
    # full_models = longcot_models + shortcot_models
    # output_dir = "/home/yxy/RLAMG/data"
    
    # openr1_multi_longcot_data,_ = get_train_data(models_list=longcot_models)
    # openr1_multi_shortcot_data,_ = get_train_data(models_list=shortcot_models)
    # openr1_multi_all_data, _ = get_train_data(models_list=full_models)
    
    # openr1_multi_longcot_data = get_common_prompt_data(openr1_multi_longcot_data, openr1_multi_shortcot_data)
    # openr1_multi_shortcot_data = get_common_prompt_data(openr1_multi_shortcot_data, openr1_multi_longcot_data)
    # openr1_multi_all_data = get_common_prompt_data(openr1_multi_all_data, openr1_multi_longcot_data)
    
    # if output_dir:
    #     save_parquet(f"{output_dir}/openr1_multi_longcot.parquet", openr1_multi_longcot_data)
    #     convert_parquet_to_json(f"{output_dir}/openr1_multi_longcot.parquet")
    #     save_parquet(f"{output_dir}/openr1_multi_shortcot.parquet", openr1_multi_shortcot_data)
    #     convert_parquet_to_json(f"{output_dir}/openr1_multi_shortcot.parquet")
    #     save_parquet(f"{output_dir}/openr1_multi_all.parquet", openr1_multi_all_data)
    #     convert_parquet_to_json(f"{output_dir}/openr1_multi_all.parquet")
    
    # print(f"数据处理完成！输出文件路径: {output_dir}")
    
    # 筛选指定模型的数据
    # specified1_models = ['Qwen3-8B_thinking']
    # specified2_models = ['Qwen3-8B_thinking', 'Qwen3-8B']
    # specified4_models = ['Qwen3-8B_thinking', 'DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B', 'Qwen2.5-Math-7B-Instruct']
    
    # input_file = "/home/yxy/RLAMG/data/openr1_multi_specified4_models.parquet"
    # output_file = "/home/yxy/RLAMG/data/openr1_multi_specified2_models.parquet"
    # filtered_data = filter_data_by_models(input_file, specified2_models, output_file)
    
    # process openr1 data
    # openr1_data = read_praquet_file("/home/yxy/RLAMG/data/openr1.parquet")
    # openr1_multi_all_data = read_praquet_file("/home/yxy/RLAMG/data/openr1_multi_all.parquet")
    # openr1_8k_data= get_common_prompt_data(openr1_data, openr1_multi_all_data)
    # save_parquet("/home/yxy/RLAMG/data/openr1-8.5k.parquet", openr1_8k_data)
    # convert_parquet_to_json("/home/yxy/RLAMG/data/openr1-8.5k.parquet")
    
    # openr1_data = '/home/yxy/RLAMG/data/openr1-8.5k.parquet'
    # longcot_data = '/home/yxy/RLAMG/data/openr1_multi_longcot.parquet'
    # # shortcot_data = "/home/yxy/RLAMG/data/openr1_multi_shortcot.parquet"
    # output_path = '/home/yxy/RLAMG/data/openr1_multi_longcot_r1.parquet'
    # # output_path = '/home/yxy/RLAMG/data/openr1_multi_shortcot_r1.parquet'
    # add_openr1_data(openr1_data, longcot_data, output_path)
    
    # valid_data = read_praquet_file('/data/RLAMG/data/valid.parquet')
    # output_path = '/data/RLAMG/data/valid_pr.parquet'
    # process_valid_data(valid_data, output_path)
    
    # math_all_data = read_praquet_file('/data/RLAMG/data/valid_math_all.parquet')
    # arc_c_data = read_praquet_file('/data/RLAMG/data/valid.arc_c.parquet')
    # gpqa_data = read_praquet_file('/data/RLAMG/data/valid.gpqa.parquet')
    # mmlu_pro_data = read_praquet_file('/data/RLAMG/data/valid.mmlu_pro.parquet')
    # all_data = math_all_data + arc_c_data + gpqa_data + mmlu_pro_data
    # output_path = '/data/RLAMG/data/valid_all.parquet'
    # process_valid_data(all_data, output_path)
    
    # valid_data = read_praquet_file('/home/yxy/RLAMG/data/openr1-8.5k.parquet')
    # output_path = '/home/yxy/RLAMG/data/openr1-8.5k.parquet'
    # replace_system_prompt(valid_data, output_path)
    # convert_parquet_to_json(f"{output_dir}/openr1_multi_longcot.parquet")
    
    # get sft data
    data = read_praquet_file('/home/yxy/RLAMG/data/openr1_multi_longcot.parquet')
    sft_data = get_sft_data(data)
    save_json('/home/yxy/RLAMG/data/openr1_multi_longcot_sft.json', data)
    