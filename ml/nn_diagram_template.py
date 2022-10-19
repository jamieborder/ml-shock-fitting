
template_nn = '''
# dot -Tpng -O nn_diagram
digraph G {{

        rankdir=LR
	    splines=line
        nodesep=.10;

        node [label=""];

        graph [nodesep=0.1, ranksep=4];

        subgraph cluster_0 {{
		color=white;
                node [style=solid,color=blue4, shape=circle];
        {FIRST_NODES};
		label = "inputs";
	}}

    {SUBGRAPHS}

    {CONNECTIONS}

}}
'''

template_sg = '''
	subgraph cluster_{ID} {{
		color=white;
		node [style=solid,color={COLOUR}, shape=circle];
        {NODES};
		label = "{LABEL}";
	}}

'''
