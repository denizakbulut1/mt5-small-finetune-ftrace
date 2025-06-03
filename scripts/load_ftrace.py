from datasets import load_dataset

queries = load_dataset("ekinakyurek/ftrace", "queries")
abstracts = load_dataset("ekinakyurek/ftrace", "abstracts")

# Print available splits
print("Queries splits:", queries.keys())
print("Abstracts splits:", abstracts.keys())

# Print a few examples from each
print("\nSample Query Example:")
print(queries["train"][0])

print("\nSample Abstract Example:")
print(abstracts["train"][0])