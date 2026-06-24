import pickle

with open(
    "dataset/IRAS_data/openval_data_new_prompt.pkl",
    "rb"
) as f:
    data = pickle.load(f)

print(type(data))
print(len(data))

target = "ue74f81c5-c851-4089-8dc0-e9944b0d4044"

for item in data:
    if target in str(item):
        print(item)
        break