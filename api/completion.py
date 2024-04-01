
def test(client, model, prompt, max_tokens):
    print("----------Completion")
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )
    print(response)


def quick_sort(arr):
    print("----------Completion.QuickSort", arr)
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) - 1]
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)