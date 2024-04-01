import pandas as pd
import tiktoken
from openai import OpenAI
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 第一步，加载数据集
input_data_path = "data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_data_path, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: "+ df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
print("--第一步，显示前两行数据--")
print(df.head(2))
print("--第一步，显示combined--")
print(df["combined"])

# 第二步，Embedding模型关键参数
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000
## 2.1 减少样本到最近1K条评论，并删除过长的样本
top_n = 1000
df = df.sort_values("Time").tail(top_n * 2)
df.drop("Time", axis=1, inplace=True)
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)

print("--第二步，len(df)--")
print(len(df))

# 第三步，生成Embeddings并保存
client = OpenAI()
def embedding_text1(text, model="text-embedding-ada-002"):
    res = client.embeddings.create(input=text, model=model)
    return res.data[0].embedding

def embedding_text2(text, model="text-embedding-ada-002"):
    return encoding.encode(text)

print("--第三步，embedding_text abc--")
print(embedding_text1("abc"))
print("--第三步，encoding.encode abc--")
print(embedding_text2("abc"))

# df["embedding"] = df.combined.apply(embedding_text)
# output_datapath = "data/fine_food_reviews_with_embeddings_1k_0331.csv"
# df.to_csv(output_datapath)
#
# e0 = df["embedding"][0]
# print(e0)

## 第四步，读取embeddings文件
embedding_data_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df_embedded = pd.read_csv(embedding_data_path, index_col=0)
print(df_embedded["embedding"])

print(len(df_embedded["embedding"][0]))
print(type(df_embedded["embedding"][0]))

## 第五步，转为向量
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)
print(len(df_embedded["embedding_vec"][0]))

print(df_embedded.head(2))

# 第五步，使用 t-SNE 可视化 1536 维 Embedding 美食评论
print(type(df_embedded["embedding_vec"]))

assert df_embedded['embedding_vec'].apply(len).nunique() == 1

matrix = np.vstack(df_embedded['embedding_vec'].values)
tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42,
    init='random',
    learning_rate=200
)

vis_dims = tsne.fit_transform(matrix)
colors = ["red","darkorange", "gold", "turquoise", "darkgreen"]

x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
color_indices = df_embedded.Score.values - 1
assert len(vis_dims) == len(df_embedded.Score.values)

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
plt.title("Amazon ratings visualized in language using t-SNE")
plt.savefig('t-sne/matrix.jpg')

# 第六步，使用K-Means聚类，然后使用t-SNE可视化
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
kmeans.fit(matrix)
df_embedded['Cluster'] = kmeans.labels_
print(df_embedded['Cluster'])
print(df_embedded.head(2))

colors = ["red", "green", "blue", "purple"]
tsne_model = TSNE(n_components=2, random_state=42)
vis_data = tsne_model.fit_transform(matrix)
x = vis_data[:, 0]
y = vis_data[:, 1]
color_indices = df_embedded['Cluster'].values
colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap)
plt.title("Clustering visualized in 2D using t-SNE")
plt.savefig('t-sne/k-means.jpg')

# 第七步，使用Embedding进行文本搜索
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(type(df_embedded["embedding_vec"][0]))


def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = embedding_text1(product_description)

    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

res = search_reviews(df_embedded, 'delicious beans', n=3)
print(res)

def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = embedding_text1(product_description)

    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

res = search_reviews(df_embedded, 'dog food', n=3)
print(res)