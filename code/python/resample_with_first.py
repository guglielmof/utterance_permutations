
n_perms = 10000000
t2perm = {}
with open("../../data/class_perms.txt", "r") as F:
    with open("../../data/class_perms_2.txt", "w") as F2:
        for l in F.readlines():
            topic, permutations = l.strip().split("\t")
            permutations = eval(permutations)

            permutas = []
            for p in permutations[1:n_perms]:
                p2 = list(p)
                p2 = [1] + p2[:p2.index(1)]+p2[p2.index(1)+1:]
                permutas.append(tuple(p2))
            t2perm[topic] = list(set([tuple(sorted(permutations[0]))] + permutas))

            print(len(t2perm[topic]))
            F2.write(f"{topic}\t{t2perm[topic]}\n")