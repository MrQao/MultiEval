import re

def parse_yes_no(answer: str):
    answer = answer.lower().strip()

    # 1. 直接以 yes/no 开头
    match = re.match(r"^(yes|no)\b", answer)
    if match:
        return match.group(1)

    # 2. 明确回答句式
    match = re.search(
        r"\b(?:the answer is|the answer would be)\s+(yes|no)\b",
        answer
    )
    if match:
        return match.group(1)

    # 3. 描述物体是否存在
    if re.search(r"\b(there is no|there are no|no visible)\b", answer):
        return "no"

    return None


