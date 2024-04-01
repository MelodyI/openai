
def test(client):
    print("----------ChatCompletion")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
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
        ]
    )
    print(response)