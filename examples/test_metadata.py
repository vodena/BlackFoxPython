from blackfox import BlackFox
import io

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

file = 'data/optimized_network_cancer.h5'

m = bf.get_metadata(file)
print(m)

with open(file, 'rb') as fin:
    data = io.BytesIO(fin.read())
    fin.seek(0)
    data2 = io.BytesIO(fin.read())

m2 = bf.get_metadata(data2)
print(m2)
