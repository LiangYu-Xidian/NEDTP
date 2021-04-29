from gensim.models import Word2Vec

fp1 = open('../walk_path/all_protein_walkpath.txt','r')
walks = []
for line in fp1:
    walk = line.strip().split('\t')
    walks.append(walk)
model = Word2Vec(walks, size=400, window=4, sg=1, min_count=10 ,workers=10, iter=40, hs=0, negative=6)
feature = open('../feature/protein_feature.txt','w')
for i in range(1916):
    if i == 0:
        continue
    for vec in model[str(i)]:
        feature.write(str(vec)+' ')
    feature.write('\n')
feature.close()
fp1.close()
