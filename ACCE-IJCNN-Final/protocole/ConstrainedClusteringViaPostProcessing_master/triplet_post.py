from gurobipy import *

# SCALE_PW = 2000
# SCALE_PW = 12000
SCALE_PW = 1
start_idx = 5
end_idx = 6
PW_ARR = [3600, 30000, 60000]
# PW_ARR = [60000]
# PW_ARR = [600, 5000, 10000]
# Maximum test = 20
TOTAL_TEST_FOR_EACH_SET = 5


def clustering_triplet(n, k, dis_matrix, triplets):
    model = Model("Clustering pairwise constraints using MCM distance")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-dis_matrix[i][j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(len(triplets)):
        a, p, n = triplets[i][0], triplets[i][1], triplets[i][2]
        for j in range(k):
            model.addConstr(LinExpr([-1, 1, -1], [x[a, j], x[p, j], x[n, j]]), ">=", -1,
                            name="Triplet[%s]Cluster[%s]" % (i, j))
    model.update()
    model.__data = x
    return model


def extract_cluster_id(n, c, k):
    ans = []
    for i in range(n):
        is_add = False
        for j in range(k):
            if c[i, j].X == 1:
                ans.append(j)
                is_add = True
        if not is_add:
            for j in range(k):
                if abs(c[i, j].X - 1) < 0.00001:
                    ans.append(j)
                    break
    return ans


def run_model_triplet(n, k, dis, triplets):
    model = clustering_triplet(n, k, dis, triplets)
    # model.write('clustering-car-pw.lp')
    model.optimize()
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
        return
    c = model.__data
    partition = extract_cluster_id(n, c, k)
    #print(partition)
    #return model.Runtime, partition
    return partition
