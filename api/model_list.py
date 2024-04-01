import tiktoken


def test(client):
    print("----------获取模型列表")
    models = client.models.list()
    for item in models:
        print("id: '{0}'".format(item.id))
        encoding = tiktoken.encoding_for_model(item.id)
        print(encoding.name.lower())
        # print(tiktoken.get_encoding(encoding_name))

