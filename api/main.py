import os
from openai import OpenAI

# 这个是PPT上的例子，已经过时了
# api_key = os.getenv("OPENAI_API_KEY")
# print(api_key)
# openai.api_key = api_key
#
# print(openai.Model.list())
#

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print(client.models.list())