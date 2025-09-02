import re
from math_verify import parse, verify

def get_last_sentence(generated_response: str) -> str:
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', 
        generated_response.strip()
    )
    last_sentence = sentences[-1] if sentences else ''
    return last_sentence.strip()
def last_boxed_only_string(string):
    """找到最后一个 \boxed 或 \fbox 及其对应的完整内容"""
    # 先尝试找 \boxed
    idx = string.rfind('\\boxed')
    if idx < 0:
        # 如果没找到，尝试找 \fbox
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    # 从找到的位置开始，计算花括号的配对
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    
    return string[idx:right_brace_idx + 1]

def remove_boxed(s):
    """去除 \boxed{...} 的外层标记，只保留内容"""
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except:
        # 尝试 \fbox 格式
        left = '\\fbox{'
        try:
            assert s[:len(left)] == left
            assert s[-1] == '}'
            return s[len(left):-1]
        except:
            return None

def extract_boxed_answer(text):
    """提取答案的主函数"""
    # 找到最后一个 boxed 内容
    boxed_str = last_boxed_only_string(text)
    if boxed_str is None:
        # return get_last_sentence(text)
        return None
    
    # 去除外层标记
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    
    # 如果答案被额外的花括号包围，去除它们
    match = re.match('^\{(.*)\}$', answer)
    if match:
        answer = match.group(1)
    
    return answer
def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer

def fix_fracs(string):

    """修复分数表示,如 \frac1b -> \frac{1}{b}"""
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string

def fix_a_slash_b(string):
    """修复 a/b 形式为 \frac{a}{b},支持更复杂的分数形式
    例如: 
    - 1/2 -> \frac{1}{2}
    - \sqrt{3}/2 -> \frac{\sqrt{3}}{2}
    - -\sqrt{3}/2 -> -\frac{\sqrt{3}}{2}
    """
    if len(string.split('/')) != 2:
        return string
        
    numerator = string.split('/')[0].strip()
    denominator = string.split('/')[1].strip()
    
    # 处理负号
    sign = ''
    if numerator.startswith('-'):
        sign = '-'
        numerator = numerator[1:]
    
    # 处理分子中的根号
    if '\\sqrt' in numerator:
        numerator = fix_sqrt(numerator)
    
    # 尝试将分子分母转为整数(如果可能的话)
    try:
        num = int(numerator)
        den = int(denominator)
        numerator = str(num)
        denominator = str(den)
    except ValueError:
        pass
        
    return f'{sign}\\frac{{{numerator}}}{{{denominator}}}'

def fix_sqrt(string):
    """修复根号表示,如 \sqrt3 -> \sqrt{3}"""
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string

def process_matrix_element(element):
    """处理矩阵元素中的分数和根号表达式"""
    element = element.strip()
    
    # 处理负号
    sign = ''
    if element.startswith('-'):
        sign = '-'
        element = element[1:]
    
    # 如果已经是\frac形式,直接返回
    if '\\frac' in element:
        return sign + element
        
    # 如果包含除号,转换为\frac形式
    if '/' in element:
        return sign + fix_a_slash_b(element)
    
    # 处理根号
    if '\\sqrt' in element:
        element = fix_sqrt(element)
    
    return sign + element

def format_latex(string):
    """格式化LaTeX字符串，统一格式"""
    # 统一行分隔符
    string = re.sub(r'\\\\+', r'\\\\', string)
    # 移除多余空格
    string = re.sub(r'\s+', '', string)
    return string

def normalize_matrix(string):
    """标准化矩阵表示"""
    # 统一矩阵类型为pmatrix
    string = string.replace('bmatrix', 'pmatrix')
    
    # 检查是否是矩阵
    if '\\begin{pmatrix}' not in string or '\\end{pmatrix}' not in string:
        return string
        
    # 提取矩阵内容
    matrix_content = string[string.find('\\begin{pmatrix}') + 15:string.find('\\end{pmatrix}')]
    
    # 处理每个矩阵元素
    # 使用正则表达式分割行，避免将其他的 \\ 也分割了
    rows = [row.strip() for row in re.split(r'\\\\(?![}])', matrix_content)]
    normalized_rows = []
    for row in rows:
        # 分割列并处理每个元素
        elements = [e.strip() for e in row.split('&')]
        normalized_elements = []
        for element in elements:
            # 确保 sqrt 前有反斜杠
            if 'sqrt' in element and '\\sqrt' not in element:
                element = element.replace('sqrt', '\\sqrt')
            # 如果元素包含分数形式，先处理分数
            if '/' in element:
                element = fix_a_slash_b(element)
            # 然后处理其他格式
            element = process_matrix_element(element)
            normalized_elements.append(element)
        normalized_rows.append('&'.join(normalized_elements))
    
    # 重新组装矩阵，确保使用正确的分隔符
    normalized_matrix = '\\begin{pmatrix}' + '\\\\'.join(normalized_rows) + '\\end{pmatrix}'
    return format_latex(normalized_matrix)  # 确保最后进行格式化

def normalize_fraction_to_decimal(string):
    """将分数表示转换为小数表示，特别处理带分数"""
    # 处理带分数形式，如 13\frac{1}{2}
    mixed_fraction_pattern = r'(\d+)\\frac\{(\d+)\}\{(\d+)\}'
    match = re.search(mixed_fraction_pattern, string)
    if match:
        whole_part = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        decimal_value = whole_part + numerator / denominator
        # 替换带分数为小数形式
        string = re.sub(mixed_fraction_pattern, str(decimal_value), string)
    
    # 处理普通分数形式，如 \frac{1}{2}
    fraction_pattern = r'\\frac\{(\d+)\}\{(\d+)\}'
    while re.search(fraction_pattern, string):
        match = re.search(fraction_pattern, string)
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        decimal_value = numerator / denominator if denominator != 0 else 0
        # 替换分数为小数形式
        string = re.sub(fraction_pattern, str(decimal_value), string, count=1)
    
    return string

def normalize_set_notation(string):
    """将集合表示法标准化，如 \{x \mid 2 < x < 3\} 转换为 (2, 3)"""
    if string is None:
        return string
    
    # 匹配集合表示法 \{x \mid a < x < b\} 或 \{x \mid a \leq x \leq b\} 等形式
    set_pattern = r'\\{x\s*\\mid\s*([\d\.-]+)\s*(<|\\leq|\\le|\\geq|\\ge|>)\s*x\s*(<|\\leq|\\le|\\geq|\\ge|>)\s*([\d\.-]+)\\}'
    match = re.search(set_pattern, string)
    
    if match:
        left_bound = match.group(1)
        left_relation = match.group(2)
        right_relation = match.group(3)
        right_bound = match.group(4)
        
        # 确定左括号类型
        left_bracket = '(' if left_relation in ['<', '\\le', '\\leq'] else '['
        
        # 确定右括号类型
        right_bracket = ')' if right_relation in ['<', '\\le', '\\leq'] else ']'
        
        # 构建标准化的区间表示
        normalized_set = f"{left_bracket}{left_bound}, {right_bound}{right_bracket}"
        
        # 替换原始表示
        string = re.sub(set_pattern, normalized_set, string)
    
    # 处理其他常见的集合表示法
    # 例如 \{x | x > a\} 转换为 (a, \infty)
    one_sided_pattern = r'\\{x\s*\\mid\s*x\s*(>|\\geq|\\ge)\s*([\d\.-]+)\\}'
    match = re.search(one_sided_pattern, string)
    if match:
        relation = match.group(1)
        bound = match.group(2)
        left_bracket = '(' if relation == '>' else '['
        normalized_set = f"{left_bracket}{bound}, \\infty)"
        string = re.sub(one_sided_pattern, normalized_set, string)
    
    # 例如 \{x | x < b\} 转换为 (-\infty, b)
    one_sided_pattern2 = r'\\{x\s*\\mid\s*x\s*(<|\\leq|\\le)\s*([\d\.-]+)\\}'
    match = re.search(one_sided_pattern2, string)
    if match:
        relation = match.group(1)
        bound = match.group(2)
        right_bracket = ')' if relation == '<' else ']'
        normalized_set = f"(-\\infty, {bound}{right_bracket}"
        string = re.sub(one_sided_pattern2, normalized_set, string)
    
    return string
def normalize_radical_expressions(string):
    """将根式表达式标准化，如 \sqrt[n]{x^m} 转换为 x^{m/n}"""
    if string is None:
        return string

    # 匹配 \sqrt[n]{x^m} 形式的表达式
    radical_pattern = r'\\sqrt\[(\d+)\]\{([a-zA-Z])\^(\d+)\}'
    
    def replace_radical(match):
        n = int(match.group(1))  # 根次
        base = match.group(2)    # 底数变量
        m = int(match.group(3))  # 指数
        
        # 计算分数指数 m/n
        numerator = m
        denominator = n
        
        # 简化分数
        from math import gcd
        g = gcd(numerator, denominator)
        if g > 1:
            numerator //= g
            denominator //= g
        
        # 如果分母为1，则直接返回整数指数
        if denominator == 1:
            return f"{base}^{numerator}"
        else:
            return f"{base}^{{{numerator}/{denominator}}}"
    
    # 替换所有匹配的表达式
    return re.sub(radical_pattern, replace_radical, string)

def normalize_expressions(string):
    """将数学表达式标准化为最简形式，如3^4 转换为 81"""
    if string is None:
        return string
    
    # 处理幂运算，如 3^4 转换为 81
    def replace_power(match):
        base = int(match.group(1))
        exponent = int(match.group(2))
        if exponent > 100:
            return match.group(0)
        try:
            return str(base ** exponent)
        except:
            return match.group(0)
    
    # 匹配纯数字的幂运算，如 3^4
    power_pattern = r'(\d+)\^(\d+)'
    string = re.sub(power_pattern, replace_power, string)
    return string

def normalize_list_format(string):
    """标准化列表格式，如 '2, 3, or 4' 转换为 '2, 3, 4'，'9 and -7' 转换为 '-7, 9'"""
    if string is None:
        return string
    
    # 移除 "or" 和 "and" 连接词，并确保用逗号分隔
    string = re.sub(r'\s+or\s+', ', ', string)
    string = re.sub(r'\s+and\s+', ', ', string)
    
    # 分割成列表并排序
    if ',' in string:
        items = [item.strip() for item in string.split(',') if item.strip()]
        # 尝试将数字项转换为数值进行排序
        try:
            # 将所有项转换为数值（整数或浮点数）
            numeric_items = []
            for item in items:
                try:
                    if '.' in item:
                        numeric_items.append(float(item))
                    else:
                        numeric_items.append(int(item))
                except ValueError:
                    # 如果无法转换为数值，保持原样
                    numeric_items.append(item)
            
            # 对可以排序的项进行排序
            sortable_items = [item for item in numeric_items if isinstance(item, (int, float))]
            non_sortable_items = [item for item in numeric_items if not isinstance(item, (int, float))]
            
            if sortable_items:
                sortable_items.sort()
            
            # 重新组合所有项，并确保没有尾随逗号
            sorted_items = sortable_items + non_sortable_items
            return ', '.join(str(item) for item in sorted_items)
        except:
            # 如果排序失败，返回原始字符串（去掉尾随逗号）
            return ', '.join(items).rstrip(',')
    
    # 如果只有一个项，直接返回（去掉尾随逗号）
    return string.strip().rstrip(',')

def strip_string(string):
    """标准化答案格式"""
    if string is None:
        return string
    
    if type(string) != str:
        string = str(string)

    string = string.strip() # 移除字符串两端的空白字符
    string = string.lower() # 将字符串转换为小写字母
    
    # 处理列表格式
    string = normalize_list_format(string)
    
    # 确保 sqrt 前有反斜杠
    if 'sqrt' in string and '\\sqrt' not in string:
        string = string.replace('sqrt', '\\sqrt')
    
    # 如果是矩阵，先标准化矩阵内容
    if '\\begin{pmatrix}' in string or '\\begin{bmatrix}' in string:
        string = normalize_matrix(string)
        return string  # 矩阵已经完全标准化，直接返回
    
    # 以下是非矩阵内容的处理
    # 移除换行符和多余空格
    string = string.replace('\n', '')
    string = string.replace('\\!', '')
    string = string.replace('\\ ', '')
    string = string.replace('  ', ' ')
    string = string.replace(' ', '')
    
    # 移除末尾句点
    string = string.rstrip('.')

    # replace \\ with \
    string = string.replace('\\', '\\')
    string = string.replace('\\\\', '\\')

    # 标准化分数表示
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    
    # 移除 \left 和 \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    
    # 移除单位
    string = string.replace('\text', '\\text')
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string
        
    # 移除角度符号
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    
    # 移除美元符号
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    
    # 移除文本标记
    string = re.sub(r'\\text\{(.*?)\}', r'\1', string)
    string = re.sub(r'\text\{(.*?)\}', r'\1', string)
    string = string.replace('x\\in', '')
    
    # 移除百分号
    string = string.replace('\\%', '')
    string = string.replace('\%', '')
    string = string.replace('%', '')
    
    # 标准化小数点
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if len(string) > 0 and string[0] == '.':
        string = '0' + string

    # 移除其他特殊字符
    string = string.replace('\\cdot', '')
    string = re.sub(r'\\mbox{.*?}', '', string)
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')
    string = string.replace("'", '')
    string = string.replace('"', '')
    string = string.replace(',', '')
    # 标准化无穷
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')
    
    # 标准化虚数单位
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')
        
    # 移除多余的小数点后零
    string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0+$', r'\1', string)

    # 移除等号前的变量名
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    if string.isdigit():
        # 将字符串转为整数再转回字符串，保留有效数字
        string = str(int(string))

    # 标准化根号和分数
    string = fix_sqrt(string)
    string = fix_fracs(string)
    string = fix_a_slash_b(string)

    string = normalize_radical_expressions(string)
    string = normalize_set_notation(string)
    string = normalize_fraction_to_decimal(string)
    string = normalize_expressions(string)

    # 最后统一格式化
    return format_latex(string)

def math_verify_answer(gold, answer):
    # 将gold和answer放入\\boxed{}中
    gold = parse(f"\\boxed{{{gold}}}")
    answer = parse(f"\\boxed{{{answer}}}")
    # print(gold)
    # print(answer)
    # Order here is important!
    return verify(gold, answer)

def is_equivalent(pred_answer, true_answer):
    """判断预测答案与真实答案是否等价"""
    if pred_answer is None and true_answer is None:
        return True
    if pred_answer is None or true_answer is None:
        return False

    try:
        # 尝试字符串标准化严格匹配
        norm_pred = strip_string(pred_answer)
        norm_true = strip_string(true_answer)
        # print(f'After strip_string:')
        # print(f'norm_pred: {norm_pred}')
        # print(f'norm_true: {norm_true}')
        if norm_true == norm_pred:
            return True
        else:
            return False
    except Exception as e:
        print(f'Error during strip_string: {str(e)}')
        
    try:
        if math_verify_answer(true_answer, pred_answer):
            return True
    except Exception as e:
        print(f"math_verify_answer error: {e}")
        return False
        
    # # 如果标准化失败,尝试直接字符串包含判断
    return true_answer in pred_answer

if __name__ == "__main__":
    pred_answer = 'The answer is 22\\frac{1}{2}^{\circ}'
    true_answer = '22.5'
    # pred_answer = '\{x \mid 2 < x < 3\}'
    # true_answer = '\\boxed{(2, 3)}'
    # pred_answer = '\\frac{15}{8}'
    # true_answer = '1.875'
    # pred_answer = '\\boxed{36\\sqrt{6}}'
    # true_answer = '\\boxed{6^{2.5}}'
    # pred_answer = '10^{-x}'
    # true_answer = '10^{-X}'
    # pred_answer = '2, 3, or 4'
    # true_answer = '2, 3, 4'
    # pred_answer = '9 and -7'
    # true_answer = '-7, 9'
    # pred_answer = '-1/9'
    # true_answer = '-0.1111111111111111'

    print(f"pred: {pred_answer}")
    print(f"true: {true_answer}")
    print(f"result: {is_equivalent(pred_answer, true_answer)}\n")