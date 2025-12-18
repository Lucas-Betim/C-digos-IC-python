from pgmpy.models import DiscreteBayesianNetwork #Importando rede 
G = DiscreteBayesianNetwork() #Criando rede

G.add_nodes_from(["a", "b"]) #ADD nos

G.add_edges_from([("a", "b"), ("b", "c")]) #ADD arestas(Nao tem problema 'c' nao existir, sera ADD automaticamente)

"a" in G  # check if node in graph

 len(G)  # number of nodes in graph

#Inicializate
  def __init__(
        self,
        ebunch: Optional[Union[nx.Graph, Iterable[Tuple[Any, Any]]]] = None,
        latents: Union[Set[Any], List[Any]] = set(),
        lavaan_str: Optional[str] = None,
        dagitty_str: Optional[str] = None,
    ) -> None:
        super(DiscreteBayesianNetwork, self).__init__(
            ebunch=ebunch,
            latents=latents,
        )
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u: Any, v: Any, w: Optional[Any] = None, **kwargs: Any) -> None:



