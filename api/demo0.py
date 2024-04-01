def test(client):
    print("----------获取模型列表")
    print(client.models.list())