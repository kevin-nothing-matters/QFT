import math, itertools, time, json, os
import numpy as np

xp = np

def kron(a,b): return xp.kron(a,b)
def eye(n): return xp.eye(n, dtype=xp.complex128)

def vn(rho, eps=1e-15):
    w = np.clip(np.real(np.linalg.eigvalsh(rho)), eps, 1.0)
    return float(np.sum(-w * np.log2(w)))

def pt2(rho4, keep):
    r = rho4.reshape(2,2,2,2)
    return np.trace(r, axis1=1, axis2=3) if keep==0 else np.trace(r, axis1=0, axis2=2)

def MI(rho4): return vn(pt2(rho4,0)) + vn(pt2(rho4,1)) - vn(rho4)

def cnot():
    U=xp.zeros((4,4),dtype=xp.complex128);U[0,0]=1;U[1,1]=1;U[3,2]=1;U[2,3]=1;return U

def embed(U2, ab, n=3):
    a,b=ab; Un=xp.zeros((2**n,2**n),dtype=xp.complex128)
    for s in range(2**n):
        bits=[(s>>(n-1-q))&1 for q in range(n)]; ip=(bits[a]<<1)|bits[b]
        for op in range(4):
            bo=bits.copy(); bo[a]=(op>>1)&1; bo[b]=op&1
            Un[sum(bo[q]<<(n-1-q) for q in range(n)),s]+=U2[op,ip]
    return Un

def haar4(rng):
    X=(rng.standard_normal((4,4))+1j*rng.standard_normal((4,4)))/np.sqrt(2)
    Q,R=np.linalg.qr(X); return Q*(np.diag(R)/np.abs(np.diag(R)))

def make_kraus(H):
    Ut=embed(H,(1,2))@embed(cnot(),(0,2))@embed(cnot(),(0,1)); K=[]
    for s in (0,1):
        A=xp.zeros((4,2),dtype=xp.complex128)
        for p in (0,1):
            col=Ut[:,(p<<2)|0]
            for c1 in (0,1):
                for c2 in (0,1): A[(c1<<1)|c2,p]=col[(s<<2)|(c1<<1)|c2]
        K.append(A)
    return K

def build_gates(depth, seed):
    rng=np.random.default_rng(seed); gates={}
    for gen in range(depth):
        for node in range(2**gen-1, 2**(gen+1)-1):
            gates[(gen,node)]=make_kraus(haar4(rng))
    return gates

def get_path(leaf, depth):
    return [(leaf>>(depth-1-g))&1 for g in range(depth)]

def compute_mi_pair(i, j, depth, gates):
    pi,pj=get_path(i,depth),get_path(j,depth)
    lca_gen=depth
    for k in range(depth):
        if pi[k]!=pj[k]: lca_gen=k; break
    rho=xp.array([[1.,0.],[0.,0.]],dtype=xp.complex128); node=0
    for gen in range(lca_gen):
        choice=pi[gen]; K=gates[(gen,node)]
        rho2=sum(A@rho@A.conj().T for A in K)
        rho=pt2(rho2,keep=choice); node=2*node+1+choice
    rho_joint=sum(A@rho@A.conj().T for A in gates[(lca_gen,node)])
    ni=2*node+1+pi[lca_gen]; nj=2*node+1+pj[lca_gen]
    for gen in range(lca_gen+1,depth):
        ci,cj=pi[gen],pj[gen]; K_i,K_j=gates[(gen,ni)],gates[(gen,nj)]; I2=eye(2)
        rho8=sum(kron(A,I2)@rho_joint@kron(A,I2).conj().T for A in K_i)
        r8=rho8.reshape(2,2,2,2,2,2)
        rho_joint=(np.trace(r8,axis1=1,axis2=4) if ci==0 else np.trace(r8,axis1=0,axis2=3)).reshape(4,4)
        rho8b=sum(kron(I2,B)@rho_joint@kron(I2,B).conj().T for B in K_j)
        r8b=rho8b.reshape(2,2,2,2,2,2)
        rho_joint=(np.trace(r8b,axis1=2,axis2=5) if cj==0 else np.trace(r8b,axis1=1,axis2=4)).reshape(4,4)
        ni=2*ni+1+ci; nj=2*nj+1+cj
    return rho_joint

def run_ultrametricity(depth=8, n_trees=20, n_triple_sample=50000,
                       results_dir="/home/3x-agent/qft/results", eps=1e-12):
    os.makedirs(results_dir, exist_ok=True)
    n_leaves = 2**depth
    all_pairs = list(itertools.combinations(range(n_leaves), 2))

    print(f"depth={depth}  leaves={n_leaves}  trees={n_trees}  triples/tree={n_triple_sample}")
    print(f"Pairs per tree: {len(all_pairs)}\n")

    all_U = []
    grand_start = time.time()

    for tree_idx in range(n_trees):
        t0 = time.time()
        gates = build_gates(depth, seed=1000+tree_idx)

        # Build full MI matrix
        mi_matrix = np.zeros((n_leaves, n_leaves))
        for k, (i,j) in enumerate(all_pairs):
            rho_ij = compute_mi_pair(i, j, depth, gates)
            mi = max(MI(rho_ij), eps)
            mi_matrix[i,j] = mi_matrix[j,i] = mi
            if (k+1) % 5000 == 0:
                print(f"  Tree {tree_idx+1:2d} | pair {k+1:6d}/{len(all_pairs)} | {time.time()-t0:.0f}s", flush=True)

        np.fill_diagonal(mi_matrix, 1.0)
        D = -np.log(np.maximum(mi_matrix, eps))
        np.fill_diagonal(D, 0)

        # Sample triples and test ultrametric inequality
        rng = np.random.default_rng(42 + tree_idx)
        sat = 0; tot = 0
        violations = []
        for _ in range(n_triple_sample):
            i,j,k = [int(x) for x in rng.choice(n_leaves, 3, replace=False)]
            dij,dik,djk = D[i,j], D[i,k], D[j,k]
            for (a,b,c) in [(dij,dik,djk),(dik,dij,djk),(djk,dij,dik)]:
                tot += 1
                if a <= max(b,c) + 1e-10:
                    sat += 1
                else:
                    violations.append(a - max(b,c))

        U = sat / tot
        all_U.append(U)
        elapsed = time.time() - t0
        print(f"  Tree {tree_idx+1:2d}: U={U:.4f}  violations={len(violations):5d}/{tot}  "
              f"max_viol={max(violations):.3f if violations else 0:.3f}  ({elapsed:.0f}s)", flush=True)

    mean_U = np.mean(all_U)
    std_U  = np.std(all_U)

    print(f"\n{'='*60}")
    print(f"ULTRAMETRICITY SUMMARY: depth={depth}, {n_trees} trees")
    print(f"{'='*60}")
    print(f"U = {mean_U:.4f} ± {std_U:.4f}")
    print(f"Per-tree: {[f'{u:.4f}' for u in all_U]}")
    print(f"Total elapsed: {time.time()-grand_start:.0f}s")

    result = {"depth": depth, "n_trees": n_trees, "n_triple_sample": n_triple_sample,
              "U_mean": float(mean_U), "U_std": float(std_U),
              "U_per_tree": [float(u) for u in all_U]}
    out = f"{results_dir}/depth{depth}_ultrametricity.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")
    return mean_U, std_U

if __name__ == "__main__":
    run_ultrametricity(depth=8, n_trees=20, n_triple_sample=50000)
