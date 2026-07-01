import linear_prog as lp
import numpy as np
import networkx as nx
import copy as cp


def remove_consecutive(L):
    #remove consecutive equal sets from L
    return [x for i,x in enumerate(L) if i==0 or L[i-1]!=x ]

def frac_coloring_2ec(graph, print_solution=False, print_dual=False):
    """
    Fractional coloring LP for red/blue edge-colored graphs.

    The graph is assumed to be undirected and to carry an edge attribute
    'col' with values 0 or 1. Here we interpret 0 as blue and 1 as red.

    The LP defines one variable X_I for each independent set I, with:
      - vertex coverage constraints: sum_{I : u in I} X_I >= 1 for every u,
      - red/blue pair constraints: for every blue edge {u,v} and red edge {x,y},
        sum_{I : {u,x} subset I} X_I + sum_{I : {v,y} subset I} X_I <= 1.
    """
    G = nx.relabel_nodes(nx.Graph(graph), lambda x: int(x))
    
    # Get all independent sets (not just maximal)
    L = list(nx.enumerate_all_cliques(nx.complement(G)))
    L.sort(key=tuple)
    L = remove_consecutive(L)
    frozenL = [frozenset(S) for S in L]

    # Extract blue and red edges, including both directions
    blue_edges_undirected = [(u, v) for u, v, data in G.edges(data=True) if data.get('col', 0) == 0]
    red_edges_undirected = [(u, v) for u, v, data in G.edges(data=True) if data.get('col', 0) == 1]
    
    # Add both directions for each edge
    blue_edges = [(u, v) for u, v in blue_edges_undirected] + [(v, u) for u, v in blue_edges_undirected]
    red_edges = [(u, v) for u, v in red_edges_undirected] + [(v, u) for u, v in red_edges_undirected]

    LP = lp.LinearProgram()
    Vars = LP.add_variables_from(frozenL)
    LP.set_objective({Vars[S]: 1 for S in frozenL}, minimize=True)

    for u in G.nodes():
        LP.add_constraint({Vars[S]: 1 for S in frozenL if u in S}, 1, '>=' )

    for u, v in blue_edges:
        for x, y in red_edges:
            coeffs = {}
            for S in frozenL:
                var = Vars[S]
                if u in S and x in S:
                    coeffs[var] = coeffs.get(var, 0) + 1
                if v in S and y in S:
                    coeffs[var] = coeffs.get(var, 0) + 1
            if coeffs:
                LP.add_constraint(coeffs, 1, '<=')

    _ = LP.solve()
    coloring = LP.makeColouring(L)
    opt_value = LP.result.fun if LP.minimize else -LP.result.fun

    if print_solution:
        sorted_L = [np.sort(list(S)) for S in frozenL]
        LP.print_result(VarNames=sorted_L, ShowNulVariables=False, convertToFrac=True)
        LP.print_coloring(coloring)
    if print_dual:
        print("Dual solution : ")
        for u, marginal in zip(G.nodes(), LP.result.ineqlin.marginals):
            print(f"{u} : {-lp.frac(marginal)}")

    return opt_value, coloring

def AC(n):
	"""return the run corresponding to the alternating cycle of length n"""
	return tuple([1] * n)

def runs_to_graph(runs, start_color=0):
    """
    Convert run-length encoding to a 2-edge-colored cycle graph.

    Example: (1,1,1,3,2,1) -> cycle graph with edge coloring
    (0,1,0,1,1,1,0,0,1) starting from color 0.
    """
    if not runs:
        return nx.Graph()
    
    coloring = []
    color = start_color
    for run in runs:
        coloring.extend([color] * run)
        color = 1 - color
    n = len(coloring)
    G = nx.cycle_graph(n)
    for i in range(n):
        G.edges[i, (i + 1) % n]['col'] = coloring[i]
    return G

def unfold_increase(runs, pos):
	"""
	Unfold by increasing run at position pos by 2 (same color).
	Type 1 unfolding: (a,b,c,...) -> (a,b+2,c,...)
	"""
	return tuple(x if j != pos else x + 2 for j, x in enumerate(runs))


def unfold_split(runs, pos, b1):
	"""
	Unfold by splitting run at position pos into b1 and (runs[pos]-b1) 
	with 2 opposite-color edges between.
	Type 2 unfolding: (a,b,c,...) -> (a,b1,2,b2,c,...) where b1+b2=b
	
	pos: index of run to split
	b1: size of first part (must be >= 1 and < runs[pos])
	
	Returns new runs tuple, or None if invalid.
	"""
	if b1 < 1 or b1 >= runs[pos]:
		return None
	b2 = runs[pos] - b1
	return tuple(list(runs[:pos]) + [b1, 2, b2] + list(runs[pos+1:]))


def all_unfold(runs):
	"""
	Generate all possible unfoldings of runs (both types).
	Type 1: increase each run by 2 (same color)
	Type 2: split each run and insert 2 edges of opposite color between the parts
	"""
	result = []
	
	# Type 1: increase each run by 2
	for pos in range(len(runs)):
		result.append(unfold_increase(runs, pos))
	
	# Type 2: split each run with opposite-color edges
	for pos in range(len(runs)):
		if runs[pos] >= 2:  # Need at least 2 to split meaningfully
			for b1 in range(1, runs[pos]):
				unfolded = unfold_split(runs, pos, b1)
				if unfolded:
					result.append(unfolded)
	
	return result

def fold(runs,i):
	"""
	Fold runs by removing the run at index i and merging its neighbors with 2 edges of opposite color.
	"""
	if sum(runs) <= 3:
		raise Exception("cannot fold clique")
	if i < 0 or i >= len(runs):
		return None  # Invalid index
	
	if runs[i] ==2 and i == 0:
		runs = (runs[-1],) + runs[:-1]
		i += 1
	
	if runs[i] ==2 and i == len(runs)-1:
		runs =  runs[1:] + (runs[0],)
		i -=1

	if runs[i] <2:
		raise Exception("cannot fold non-monochromatic vertex")
	elif runs[i] ==2:
		return runs[:i-1] + (runs[i-1]+runs[i+1],) + runs[i+2:]
	else:
		return runs[:i] + (runs[i]-2,) + runs[i+1:]

def run_to_coloring(runs,start_color=0):
	# return the coloring associated with the run
	if not runs:
		return []
	coloring = []
	color = start_color
	for run in runs:
		coloring.extend([color] * run)
		color = 1 - color
	return coloring

def get_canonical_coloring(coloring, n):
    """
    Get canonical form of edge coloring under rotation, reflection, and color swap.
    Returns the lexicographically smallest equivalent coloring.
    """
    candidates = []
    
    # All rotations
    for shift in range(n):
        rotated = tuple(coloring[(i - shift) % n] for i in range(n))
        candidates.append(rotated)
        # And with colors swapped
        candidates.append(tuple(1 - c for c in rotated))
    
    # All reflections (reverse direction)
    reversed_coloring = coloring[::-1]
    for shift in range(n):
        rotated = tuple(reversed_coloring[(i - shift) % n] for i in range(n))
        candidates.append(rotated)
        # And with colors swapped
        candidates.append(tuple(1 - c for c in rotated))
    
    return min(candidates)

def runs_canonical_form(runs):
	"""
	Return the maximal lexicographic rotation of the tuple `runs`.
	"""
	if not runs:
		return ()
	best = runs
	n = len(runs)
	for i in range(1, n):
		candidate = runs[i:] + runs[:i]
		if candidate > best:
			best = candidate
	return best

def join_runs_graph(run1, pos1, run2, pos2):
	"""
	Return an nx.Graph obtained by joining the graphs for run1 and run2,
	identifying vertex at index pos1 of run1 with vertex at index pos2 of run2.

	Assumes runs_to_graph(run) returns a NetworkX Graph whose nodes can be
	sorted to give a consistent ordering (0..n-1-like). pos1/pos2 are
	0-based positions; they are taken modulo the number of vertices of the run.
	"""
	g1 = runs_to_graph(run1).copy()
	g2 = runs_to_graph(run2).copy()

	n1 = g1.number_of_nodes()
	n2 = g2.number_of_nodes()
	if n1 == 0:
		return g2.copy()
	if n2 == 0:
		return g1.copy()

	nodes1 = sorted(g1.nodes())
	nodes2 = sorted(g2.nodes())

	node1 = nodes1[pos1 % n1]
	node2 = nodes2[pos2 % n2]

	# Relabel g2 nodes to avoid name collisions (start from n1)
	mapping = {old: i + n1 for i, old in enumerate(nodes2)}
	g2_relabeled = nx.relabel_nodes(g2, mapping)
	node2_new = mapping[node2]

	# Combine graphs and merge node2_new into node1
	G = nx.compose(g1, g2_relabeled)

	# Rewire neighbors of node2_new to node1, keeping the edges attributes, then remove node2_new
	for neighbor in g2_relabeled.neighbors(node2_new):
		edge_data = g2_relabeled.get_edge_data(node2_new, neighbor)
		if G.has_edge(node1, neighbor):
			if 'col' not in G[node1][neighbor] and 'col' in edge_data:
				G[node1][neighbor]['col'] = edge_data['col']
		else:
			G.add_edge(node1, neighbor, **edge_data)
	G.remove_node(node2_new)

	return G

def remove_duplicate_runs(l_runs, l_seen_before=None):
	"""
	Remove duplicate run patterns from l_runs.
	Also remove any run whose canonical coloring already appears in l_seen_before.
	"""
	if l_seen_before is None:
		l_seen_before = []
	seen_canonical = set()
	l_runs_no_duplicate = []

	for run in l_seen_before:
		coloring = run_to_coloring(run)
		canonical = get_canonical_coloring(coloring, len(coloring))
		seen_canonical.add(canonical)

	for run in l_runs:
		coloring = run_to_coloring(run)
		canonical = get_canonical_coloring(coloring, len(coloring))
		if canonical not in seen_canonical:
			seen_canonical.add(canonical)
			l_runs_no_duplicate.append(run)

	return l_runs_no_duplicate

def monochromatic_vertices(runs):
	"""
	Given a run, return the set of positions of monochromatic vertices 
	(vertices with both adjacent edges having the same color).
	"""
	if not runs:
		return set()
	
	coloring = run_to_coloring(runs)
	n = len(coloring)
	monochromatic = set()
	
	for vertex in range(n):
		# Get the colors of edges incident to this vertex
		# In a cycle, vertex i has edges to (i-1) mod n and i
		prev_edge_color = coloring[(vertex - 1) % n]
		curr_edge_color = coloring[vertex]
		
		if prev_edge_color == curr_edge_color:
			monochromatic.add(vertex)
	
	return monochromatic

def compositions(total):
	"""Yield all compositions of `total` into positive integers as tuples."""
	if total == 0:
		return
	def _comp(k):
		if k == 1:
			yield (1,)
			return
		for first in range(1, k+1):
			if first == k:
				yield (k,)
			else:
				for tail in _comp(k-first):
					yield (first,)+tail
	yield from _comp(total)

def all_runs(total):
	"""Yield all runs meaning compositions with even length and monochromatic runs."""
	for comp in compositions(total):
		if (len(comp) % 2 == 0 or len(comp) == 1) and comp not in [(1,),(2,),(1,1)]:  # Allow single run as well
			yield comp

def rotations(run):
	"""Yield all rotations of a run tuple."""
	n = len(run)
	for i in range(n):
		yield run[i:]+run[:i]

def build_fold_graph(n):
	nodes = set()
	# enumerate all runs (compositions) with total size <= n
	for total in range(1, n+1):
		for comp in all_runs(total):
			nodes.add(runs_canonical_form(comp))
	# ensure set is stable list
	nodes = set(nodes)

	G_fold = nx.DiGraph()
	G_fold.add_nodes_from(nodes)

	for u in list(nodes):
		# consider all rotations of a representative of the equivalence class
		rep = u
		for rot in rotations(rep):
			L = len(rot)
			for i in range(L):
				try:
					v = fold(rot, i)
				except Exception:
					continue
				if v is None or v not in nodes:
					continue
				v_canon = runs_canonical_form(v)
				# only include targets of size <= n
				if sum(v_canon) <= n:
					G_fold.add_node(v_canon)
					G_fold.add_edge(u, v_canon)
	return G_fold

def is_safe(runs):
	""" test if chi2ec <= 3.5 for each join of AC12 to a monochromatic vertex of runs """
	G = runs_to_graph(runs)
	val, _ = frac_coloring_2ec(G,print_solution=False)
	if val > 3.5000000001:
		return False, -1
	for pos in monochromatic_vertices(runs):
		lemniscat = join_runs_graph(AC(12), 0, runs, pos)
		val, _ = frac_coloring_2ec(lemniscat,print_solution=False)
		if val > 3.5000000001:
			return False, pos
	return True, None





if __name__ == "__main__":
	print(f"___Begin___")
	n = 18
	fold_graph = build_fold_graph(n)
	sinks = [node for node in fold_graph.nodes() if fold_graph.out_degree(node) == 0]
	print(f"Fold Graph    nodes:{fold_graph.number_of_nodes()}, edges:{fold_graph.number_of_edges()}, sinks:{len(sinks)}")

	to_explore = cp.copy(sinks)
	safe_runs = []
	unsafe_runs = []

	print(f"Starting exploration (it should take about 2 hours)")
	while to_explore:
		runs = to_explore.pop()
		
		if 'safe' in fold_graph.nodes(data=True)[runs]:
			safe = fold_graph.nodes()[runs]['safe']
			pos = None
		else:
			safe, pos = is_safe(runs)
			
		if safe:
			safe_runs.append(runs)
			print(f"✅ Safe: {runs} (size {sum(runs)})")
		else:
			# If not safe, add all predecessors to todo for further checking
			predecessors = list(fold_graph.predecessors(runs))
			to_explore.extend(predecessors)
			remove_duplicate_runs(to_explore, safe_runs)  # Remove duplicates and already safe runs
			unsafe_runs.append((runs, pos))
			val, _ = frac_coloring_2ec(runs_to_graph(runs),print_solution=False)
			if val > 3.5000000001:
				print(f"❌ Bad: {runs} (size {sum(runs)}) chi = {val} > 7/2")
			else:
				print(f"⚠️ Unsafe: {runs} (size {sum(runs)})")
	print(f"Exploration finished")

	print(f"List of safe runs: {len(safe_runs)}")
	L = [runs_canonical_form(run) for run in safe_runs]
	L.sort(key=lambda x: (sum(x),len(x), x))  # Sort by size and then lexicographically
	for run in L:
		if run == AC(len(run)):
			print(f"{run},    size {sum(run)}  (alternating cycle)")
		else:
			print(f"{run},    size {sum(run)}")
	
	print(f"____End____")