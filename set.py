A = [1,2,3,5]
B = [6,7,9,2,3]
C = {'a':1, 'b':2, 'c':3, 'd':4}
D = {'e':1, 'f':2, 'g':3, 'i':4}
AuB = set(A).union(B)
CuD = set(C).union(D)
print(CuD)
print(AuB)
print(len(AuB))

lenAuB = len(A) + len(B) - (len(set(A).intersection(B)))
print(set(A).intersection(B))

print(lenAuB)