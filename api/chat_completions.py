
def test(client, model, messages):
    print("----------ChatCompletion")
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    print(response)