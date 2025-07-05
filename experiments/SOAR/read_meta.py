import pickle

def read_pkl_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Example usage:
data = read_pkl_file('/home/sheldonvon/Proj/PCR/SOAR/dataset/metadata/train.pkl')
print(data)