import os
from openai import OpenAI

import model_list
import chat_completions
import completion

if __name__ == '__main__':
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    model_list.test(client)
    # ChartCompletion
    chat_completions.test(client, model="gpt-3.5-turbo", messages=[
        {
            "role":"system",
            "content":"You are a helpful assistant."
        },
        {
            "role":"user",
            "content":"Who won the world series in 2020?"
        },
        {
            "role":"assistant",
            "content":"The Los Angeles Dodgers won the World Series  "
        },
        {
            "role":"user",
            "content":"Where was it played? "
        }
    ])
    # Completion
    completion.test(client,
        model="gpt-3.5-turbo-instruct",
        prompt="Hello",
        max_tokens=1000
    )
    # 排序
    print(completion.quick_sort(arr=[1, 6, 3, 8, 2, 5]))


# 如何用向量数据库节省资源