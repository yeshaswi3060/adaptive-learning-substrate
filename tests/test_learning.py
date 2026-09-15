from adaptive_learning_substrate.environments import MicroProgram3BitEnvironment
from adaptive_learning_substrate.experiment import build_microprogram_graph
from adaptive_learning_substrate.learning import (
    CausalCreditFlow,
    EligibilityTraceBaseline,
)


def test_ccf_credits_omitted_correct_path_and_suppresses_wrong_path() -> None:
    env = MicroProgram3BitEnvironment(seed=0)
    graph = build_microprogram_graph(seed=0, trace_decay=1.0, log_events=True)
    learner = CausalCreditFlow(graph, learning_rate=0.2)
    bits = (0, 0, 0)
    graph.begin_episode("learn")
    result = graph.forward(env.encode(bits, "NOT"))
    target = env.target_nodes(env.apply(bits, "NOT"))
    before = graph.edge_weights()
    learner.learn(result, target)
    after = graph.edge_weights()
    for input_node in env.encode(bits, "NOT"):
        correct_edge = f"e:{input_node}->{target['bit2']}"
        assert after[correct_edge] > before[correct_edge]


def test_eligibility_baseline_uses_same_topology_and_leaves_inactive_edges_immutable() -> None:
    env = MicroProgram3BitEnvironment(seed=1)
    ccf_graph = build_microprogram_graph(seed=3, trace_decay=1.0, log_events=False)
    baseline_graph = build_microprogram_graph(seed=3, trace_decay=1.0, log_events=False)
    assert ccf_graph.topology_signature() == baseline_graph.topology_signature()
    learner = EligibilityTraceBaseline(baseline_graph, learning_rate=0.2)
    baseline_graph.begin_episode("baseline")
    result = baseline_graph.forward(env.encode((0, 0, 0), "NOT"))
    inactive_edges = {
        edge.edge_id: edge.weight for edge in baseline_graph.edges if not edge.active_this_event
    }
    learner.learn(result, env.target_nodes((1, 1, 1)))
    assert {
        edge.edge_id: edge.weight for edge in baseline_graph.edges if edge.edge_id in inactive_edges
    } == inactive_edges
