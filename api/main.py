import os
from openai import OpenAI

import demo0
import demo1

if __name__ == '__main__':
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    demo0.test(client)
    demo1.test(client)

# 如何用向量数据库节省资源