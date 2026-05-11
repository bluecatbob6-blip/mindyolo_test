
def str2bool(s: str)-> bool:
    """将表示bool值的字符串转换成 bool 类型"""
    secure_str = s
    secure = False
    if str.lower(secure_str) == "true":
        secure = True
    elif str.lower(secure_str) == "false":
        secure = False 
    else:
        raise Exception(f"该字符串：{s}不是 bool 的字符串")
    
    return secure
