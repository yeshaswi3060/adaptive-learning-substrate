# Causal Credit Flow v0 (CCF-v0)

**Status:** frozen candidate mechanism for Stage 1  
**Purpose:** make the first idea precise enough to implement, test, reject, or revise  
**Claim level:** experimental hypothesis only; no novelty or superiority claim

---

## 1. One-sentence definition

CCF-v0 is a local, event-driven learning rule in which an active unit records the
effect of omitting each participating input, and a later bounded credit packet is
divided among only those recorded inputs to update their weights and travel to
their recorded parent events.

In simpler words: a unit remembers **which recent messages actually changed its
output**. When a later outcome says "increase" or "decrease," only those remembered
messages receive a share of the signal.

---

## 2. Research boundary

### 2.1 What CCF-v0 is meant to test

CCF-v0 tests this narrow hypothesis:

> A delayed output-level correction can produce useful learning when it is routed
> through a sparse causal event history using local counterfactual omission effects,
> without automatic differentiation or a global backward pass over every operation.

The word **causal** has an operational meaning in this document. For an incoming
message on one edge, the receiving unit recomputes its own activation with that one
message omitted. The difference is the message's local omission effect. This is a
local counterfactual contribution measure; it is not proof of philosophical or
interventional causality in the wider environment.

### 2.2 What CCF-v0 includes

- a fixed sparse directed graph;
- scalar, timestamped, event-driven messages;
- strictly positive edge delays;
- a bounded nonlinear activation;
- immutable short-lived causal event records;
- an externally created, output-level signed credit signal;
- local omission effects;
- local weight changes;
- backward routing through recorded parent-event identifiers only;
- deterministic clipping, expiry, and termination rules.

### 2.3 What CCF-v0 deliberately does not include

- growth, pruning, rewiring, or module creation;
- Transformer attention;
- an autograd engine;
- a global loss-gradient traversal;
- replay of past training examples during an update;
- learned bias values;
- vector-valued messages;
- a claim of biological plausibility;
- a claim of novelty, AGI, general reasoning, or state-of-the-art performance.

Topology adaptation is postponed on purpose. If the local learning rule cannot pass
simple fixed-graph tests, growth and pruning would only add a confound and make the
failure harder to diagnose.

---

## 3. Formal system

### 3.1 Fixed sparse directed graph

Let

\[
G=(V,E)
\]

be a directed graph with units \(V\) and edges \(E\). An edge is written

\[
e=(i\rightarrow j),
\]

where \(i\) is the source unit and \(j\) is the destination unit.

The graph is **sparse**: \(|E|\ll |V|^2\). Every experiment must record the exact
edge list, maximum in-degree, maximum out-degree, and random seed. The edge list is
fixed for the full CCF-v0 run. Enabling, disabling, adding, or deleting an edge is
not allowed during learning.

The structural graph may contain cycles, but every edge has an integer delay

\[
d_e\ge 1.
\]

Therefore, a message emitted at tick \(t\) can arrive no earlier than tick
\(t+1\). Even if the structural graph is recurrent, its realized event ancestry is
acyclic because every parent event has an earlier timestamp than its child event.

### 3.2 Numeric domain

The reference implementation uses IEEE-754 `float64`. Unless an experiment freezes
another activation before any result is observed, every non-input unit uses

\[
\phi(z)=\tanh(z).
\]

Input/source units create timestamped source events directly. A source event has no
parents and is therefore the natural end of credit routing.

### 3.3 Notation

| Symbol | Meaning |
|---|---|
| \(i,j\) | source and destination unit identifiers |
| \(e=(i\to j)\) | a directed edge |
| \(w_e\) | current scalar weight on edge \(e\) |
| \(d_e\) | positive integer transmission delay |
| \(b_j\) | fixed bias of unit \(j\) |
| \(t\) | discrete event tick |
| \(x_e(t)\) | scalar payload that arrives through edge \(e\) at tick \(t\) |
| \(u_j(t)\) | preactivation of unit \(j\) |
| \(a_j(t)\) | activation computed by unit \(j\) |
| \(o_e(t)\) | omission effect of edge message \(e\) |
| \(g_e(t)\) | local secant direction for changing \(w_e\) |
| \(h_e(t)\) | local secant direction for changing the source payload |
| \(q_e(t)\) | nonnegative share of the arriving unit credit assigned to \(e\) |
| \(c_j\) | signed activation-coordinate credit arriving at unit event \(j\) |
| \(T\) | tick at which feedback becomes available |
| \(T-t\) | age of a causal record when feedback arrives |

---

## 4. State and public records

The following names are part of the CCF-v0 public interface and should be used by
the simulator, tests, and experiment logs.

### 4.1 `EdgeState`

Each edge stores only:

```text
EdgeState:
    edge_id
    source_unit_id
    destination_unit_id
    weight              # mutable scalar
    delay_ticks          # fixed integer >= 1
```

The endpoints and delay are immutable in v0. Only `weight` can learn.

### 4.2 `UnitState`

Each non-input unit stores:

```text
UnitState:
    unit_id
    bias                # fixed in v0
    last_emitted_value
    next_event_sequence
    trace_buffer        # bounded collection of UnitEvent records
```

The initial `last_emitted_value` is zero. Bias is fixed; an experiment that needs
a learnable offset must provide a constant-valued source unit and an ordinary edge.
That keeps every learned parameter inside the same credit rule.

### 4.3 `EventMessage`

An emitted message contains:

```text
EventMessage:
    parent_event_id
    edge_id
    value
    arrival_tick
```

For one destination unit, all messages with the same `arrival_tick` are processed
together. A source unit may emit at most one message per outgoing edge per tick.

### 4.4 `EdgeTrace`

For every nonzero message used by an emitted unit event, the destination stores:

```text
EdgeTrace:
    edge_id
    parent_event_id
    message_value       # x
    omission_effect     # o
    weight_secant       # g
    source_secant       # h
```

All values are computed during the forward event and are then immutable. Later
weight changes must not rewrite old traces.

### 4.5 `UnitEvent`

```text
UnitEvent:
    event_id            # unique (episode_id, unit_id, sequence)
    episode_id
    unit_id
    tick
    preactivation       # u
    activation          # a
    forced_output       # boolean
    edge_traces[]       # local incoming EdgeTrace records
```

A `UnitEvent` is retained for at most `H_trace` ticks. An evicted event cannot be
reconstructed from the current graph and cannot receive credit.

### 4.6 `CreditPacket`

```text
CreditPacket:
    credit_id           # identifies one external feedback occurrence
    destination_event_id
    signed_credit       # c
    feedback_tick       # T
    hop_count
```

Credit packets reference event identifiers, not just unit identifiers. This prevents
a delayed outcome from being assigned to a later, unrelated activation of the same
unit.

---

## 5. Forward event rule

### 5.1 Aggregate only messages that actually arrive

Let \(P_j(t)\) be the set of event messages arriving at unit \(j\) at tick \(t\).
If \(P_j(t)\) is empty and the unit is not explicitly queried as an output, the unit
does nothing at that tick.

When \(P_j(t)\) is nonempty, the unit computes

\[
u_j(t)=b_j+\sum_{e\in P_j(t)}w_e x_e(t)
\]

and

\[
a_j(t)=\phi\!\left(u_j(t)\right).
\]

This is a local operation: unit \(j\) needs only its bias, its incoming messages,
and the weights of those incident edges.

### 5.2 Sparse emission gate

The unit emits if

\[
\operatorname{emit}_j(t)=
\Bigl(|a_j(t)-\ell_j|\ge \theta_{\mathrm{emit}}\Bigr)
\;\lor\;
\operatorname{forcedOutput}_j(t),
\]

where \(\ell_j\) is the unit's last emitted value.

If the gate is false, no `UnitEvent` is retained and no downstream message is sent.
If the gate is true:

1. the unit creates a `UnitEvent`;
2. it sets \(\ell_j\leftarrow a_j(t)\);
3. it sends payload \(a_j(t)\) through each outgoing edge;
4. an outgoing edge \(e'\) schedules that payload for tick \(t+d_{e'}\).

Output queries are forced because a measured decision needs a stable event identifier
to which feedback can later be attached.

### 5.3 Local counterfactual omission effect

For one arrived edge message \(e=(i\to j)\), unit \(j\) computes a local alternate
activation in which only that message is removed:

\[
a_{j\setminus e}(t)
=
\phi\!\left(u_j(t)-w_e x_e(t)\right).
\]

The omission effect is

\[
\boxed{
o_e(t)=a_j(t)-a_{j\setminus e}(t)
}
\tag{1}
\]

Plain-language interpretation:

- \(o_e>0\): including this message raised the unit's activation;
- \(o_e<0\): including this message lowered the unit's activation;
- \(o_e\approx 0\): omitting it made almost no local difference.

This requires one additional local activation-function evaluation per nonzero arrived
message. Those evaluations must be counted in the compute budget.

### 5.4 Two local secants derived from the omission effect

The weight secant is

\[
\boxed{
g_e(t)=
\begin{cases}
o_e(t)/w_e, & |w_e|\ge \epsilon_w,\\
0, & |w_e|<\epsilon_w.
\end{cases}
}
\tag{2}
\]

It estimates the direction in which changing the edge weight changes the current
unit activation, using the finite change already measured by omission. For a linear
unit, \(g_e=x_e\) exactly.

The source secant is

\[
\boxed{
h_e(t)=
\begin{cases}
o_e(t)/x_e(t), & |x_e(t)|\ge \epsilon_x,\\
0, & |x_e(t)|<\epsilon_x.
\end{cases}
}
\tag{3}
\]

It estimates the sign of the effect that a change in the source payload would have
on this unit. For a linear unit, \(h_e=w_e\) exactly.

No derivative is obtained from another unit and no chain-rule product is stored.
Both values use only the destination unit's actual activation, its one-message
omission activation, and the local edge/message values.

An edge with \(|w_e|<\epsilon_w\), a message with
\(|x_e|<\epsilon_x\), or exactly zero omission effect receives no useful v0 trace.
CCF-v0 has no hidden bootstrap update for such a dormant edge. All initialized
active edges must therefore start with \(|w_e|\ge\epsilon_w\). Whether this
"dead-edge" behavior is a weakness is an experimental question, not something to
silently repair after observing results.

### 5.5 The trace is stored only when the unit emits

For an emitted unit event, Equations (1)-(3), the incoming message value, and its
parent event identifier are stored in an `EdgeTrace`. The trace is immutable and
expires after `H_trace` ticks.

This record is the complete path information available to delayed CCF credit. CCF
must never reconstruct a path later by searching the current graph.

---

## 6. Meaning and creation of a root credit signal

The CCF core accepts a bounded signed scalar in the coordinate system of a recorded
output activation:

\[
c_o>0 \quad\text{means "increase this recorded output activation,"}
\]

\[
c_o<0 \quad\text{means "decrease this recorded output activation."}
\]

For a supervised scalar target \(y_o\), the default adapter is

\[
\boxed{
c_o=\operatorname{clip}(y_o-a_o,-C_{\max},C_{\max}).
}
\tag{4}
\]

This resembles a prediction error, but it is injected only at the designated output
event. The controller does not calculate hidden-unit errors.

A success/failure or reinforcement-learning experiment needs a separately frozen
output adapter that converts the observed outcome into this same signed activation
coordinate. That adapter is outside the CCF-v0 core and must be preregistered. It is
not acceptable to change the adapter after results in order to rescue the rule.

The environment/controller may know the target, reward, or terminal outcome only to:

1. calculate the root scalar credit;
2. attach it to one or more explicitly designated output `UnitEvent` identifiers;
3. compute evaluation metrics.

It may not create hidden credits or reveal the target during the hidden forward
events.

---

## 7. Delayed credit allocation, update, and routing

Suppose feedback with identifier \(k\) becomes available at tick \(T\), and an
aggregated credit \(c_j\) reaches a retained `UnitEvent` of unit \(j\) created at
tick \(t\).

### 7.1 Aggregate reconvergent packets first

Several downstream event paths can point to the same parent event. For a single
`credit_id`, all packets addressed to one `event_id` are summed before that event is
processed:

\[
c_j=
\operatorname{clip}\!\left(
\sum_{p\,\to\,\text{same event}}c_p,
-C_{\max},C_{\max}
\right).
\tag{5}
\]

Processing uses descending event time. Positive edge delays guarantee that every
child is processed before its parents, so all reconvergent packets are present when
the parent event is processed.

### 7.2 Divide credit by recorded omission magnitude

For the incoming traces stored by this event, form the local denominator

\[
D_j(t)=\epsilon_{\mathrm{route}}+
\sum_{r\in P_j(t)}|o_r(t)|.
\tag{6}
\]

The share assigned to edge trace \(e\) is

\[
\boxed{
q_e(t)=\frac{|o_e(t)|}{D_j(t)}.
}
\tag{7}
\]

The signed credit share for that edge is

\[
\boxed{
c_e=c_j q_e(t).
}
\tag{8}
\]

The shares are nonnegative and satisfy

\[
0\le q_e<1,
\qquad
\sum_e q_e\le 1.
\]

Thus a message that caused a larger local omission effect receives a larger fraction
of the unit's credit. If every omission effect is zero, every share is zero and the
packet stops at this event.

Normalization is only across the participating incoming traces of **one unit event**.
There is no graph-wide normalization.

### 7.3 Apply age decay and update the edge

The age of this trace at feedback is

\[
\Delta t=T-t.
\]

The unbounded local proposal is

\[
\widetilde{\Delta w_e}
=
\eta\,
\lambda_{\mathrm{age}}^{\Delta t}\,
c_e\,g_e(t).
\tag{9}
\]

The actual per-participation update is

\[
\boxed{
\Delta w_e=
\operatorname{clip}\!\left(
\widetilde{\Delta w_e},
-\Delta_{\max},
\Delta_{\max}
\right)
}
\tag{10}
\]

followed by

\[
\boxed{
w_e\leftarrow
\operatorname{clip}(w_e+\Delta w_e,-W_{\max},W_{\max}).
}
\tag{11}
\]

All proposals for one `UnitEvent` are calculated from its frozen traces before any
of that event's edge weights are mutated. They are then committed in ascending
`edge_id` order. If the same edge appears in several retained events for the same
feedback occurrence, it can receive one update per causal participation, processed
in descending event time.

The factor \(\lambda_{\mathrm{age}}^{\Delta t}\) weakens stale evidence. It does
not cause an old record to survive beyond the hard trace horizon.

### 7.4 Route a packet to the recorded parent event

After allocating \(c_e\), the packet routed to the exact recorded parent event is

\[
\boxed{
c_{\operatorname{parent}(e)}
\;\mathrel{+}=\;
\gamma_{\mathrm{route}}\,
c_e\,
\operatorname{sign}\!\left(h_e(t)\right).
}
\tag{12}
\]

The magnitude is the allocated causal share. The sign of \(h_e\) converts a desired
change in the destination activation into the corresponding desired direction for
the source event value.

The parent event is identified only by `parent_event_id` stored in the trace. The
runtime must not route to "the most recent event from unit i" or scan the graph for
an alternative path if the recorded parent is missing.

### 7.5 Stop conditions

A branch stops immediately when any of these conditions is true:

- the current record does not exist or has expired;
- \(T-t>H_{\mathrm{trace}}\);
- `hop_count >= H_hop`;
- \(|c_j|<c_{\min}\);
- the event is a source event with no parent traces;
- all local omission effects are zero;
- a referenced parent record is unavailable.

Every stop reason must be counted in the run log.

---

## 8. Why the update direction is locally sensible

Consider one monotone unit with one active incoming edge, away from clipping and
saturation. If the arriving credit is positive, Equation (9) changes the weight in
a direction that locally increases the recorded output on replay.

For monotone \(\phi\):

\[
\operatorname{sign}(o_e)=\operatorname{sign}(w_e x_e).
\]

Therefore

\[
\operatorname{sign}(g_e)
=
\operatorname{sign}(o_e/w_e)
=
\operatorname{sign}(x_e).
\]

For \(c_e>0\), the preactivation change on replay is proportional to

\[
x_e\Delta w_e>0.
\]

For \(c_e<0\), it is negative. This is a required one-edge sanity property. It
does not prove that a multi-step network will learn a useful global behavior; that
is what the experiments must determine.

---

## 9. Numerical worked example

This example uses the frozen default \(\tanh\) activation and shows every number.
Suppose unit \(j\) receives two messages at tick \(t\):

| Quantity | Edge 1 | Edge 2 |
|---|---:|---:|
| message \(x_e\) | \(2.0\) | \(-1.0\) |
| weight \(w_e\) | \(0.3\) | \(0.4\) |

Let the fixed bias be \(b_j=0.1\). Then

\[
u_j=0.1+(0.3)(2.0)+(0.4)(-1.0)=0.3
\]

and

\[
a_j=\tanh(0.3)=0.2913126125.
\]

### 9.1 Omit Edge 1

Without Edge 1,

\[
u_{j\setminus 1}=0.3-(0.3)(2.0)=-0.3,
\]

so

\[
o_1
=
\tanh(0.3)-\tanh(-0.3)
=
0.5826252249.
\]

The two secants are

\[
g_1=o_1/w_1=1.9420840830,
\]

\[
h_1=o_1/x_1=0.2913126125.
\]

### 9.2 Omit Edge 2

Without Edge 2,

\[
u_{j\setminus 2}=0.3-(0.4)(-1.0)=0.7,
\]

so

\[
o_2
=
\tanh(0.3)-\tanh(0.7)
=
-0.3130551647.
\]

The two secants are

\[
g_2=o_2/w_2=-0.7826379117,
\]

\[
h_2=o_2/x_2=0.3130551647.
\]

Notice the signs. Edge 2's current negative message lowered the output, so its
omission effect is negative. However, increasing the **source value** on that edge
would raise the output, so \(h_2\) is positive.

### 9.3 Delayed positive credit

Assume feedback arrives two ticks later with

\[
c_j=+0.5,
\quad
\eta=0.01,
\quad
\lambda_{\mathrm{age}}=0.97,
\quad
\gamma_{\mathrm{route}}=0.9.
\]

Ignoring the \(10^{-12}\) route epsilon at the shown precision,

\[
D_j=|0.5826252249|+|-0.3130551647|=0.8956803896.
\]

Therefore

\[
q_1=0.6504833998,
\qquad
q_2=0.3495166002,
\]

and

\[
c_1=(0.5)q_1=0.3252416999,
\]

\[
c_2=(0.5)q_2=0.1747583001.
\]

The age factor is

\[
0.97^2=0.9409.
\]

No delta clip is reached, so

\[
\Delta w_1
=(0.01)(0.9409)(0.3252416999)(1.9420840830)
=0.0059431641,
\]

\[
\Delta w_2
=(0.01)(0.9409)(0.1747583001)(-0.7826379117)
=-0.0012868922.
\]

The new weights are

\[
w_1'=0.3059431641,
\qquad
w_2'=0.3987131078.
\]

If the same messages are replayed only for this arithmetic check, the new
preactivation and activation are

\[
u_j'=0.1+(0.3059431641)(2.0)+(0.3987131078)(-1.0)
=0.3131732203,
\]

\[
a_j'=\tanh(0.3131732203)=0.3033211356.
\]

The activation rose from \(0.2913126125\) to \(0.3033211356\), matching the
meaning of positive credit.

The routed parent credits are

\[
c_{p_1}=(0.9)(0.3252416999)\operatorname{sign}(h_1)
=0.2927175299,
\]

\[
c_{p_2}=(0.9)(0.1747583001)\operatorname{sign}(h_2)
=0.1572824701.
\]

Both are positive because both source secants are positive.

The replay calculation above is only a unit-test explanation. The CCF update itself
does not replay the stored input.

---

## 10. Reference pseudocode

### 10.1 Forward event processing

```text
function PROCESS_ARRIVALS(unit j, tick t, messages, forced_output=false):
    assert every message.arrival_tick == t
    assert at most one message per incoming edge

    u = j.bias
    for message m in messages sorted by m.edge_id:
        e = EDGE[m.edge_id]
        u = u + e.weight * m.value

    a = tanh(u)
    should_emit = forced_output or abs(a - j.last_emitted_value) >= theta_emit

    if not should_emit:
        return NO_EVENT

    traces = []
    for message m in messages sorted by m.edge_id:
        e = EDGE[m.edge_id]

        if abs(m.value) < eps_x:
            continue

        omitted_a = tanh(u - e.weight * m.value)
        o = a - omitted_a
        g = o / e.weight if abs(e.weight) >= eps_w else 0.0
        h = o / m.value   if abs(m.value) >= eps_x else 0.0

        traces.append(EdgeTrace(
            edge_id=e.edge_id,
            parent_event_id=m.parent_event_id,
            message_value=m.value,
            omission_effect=o,
            weight_secant=g,
            source_secant=h
        ))

    event = UnitEvent(
        event_id=NEXT_EVENT_ID(j),
        episode_id=CURRENT_EPISODE,
        unit_id=j.unit_id,
        tick=t,
        preactivation=u,
        activation=a,
        forced_output=forced_output,
        edge_traces=IMMUTABLE(traces)
    )

    j.trace_buffer.insert(event)
    j.trace_buffer.evict_older_than(t - H_trace)
    j.last_emitted_value = a

    for outgoing edge e_out sorted by e_out.edge_id:
        schedule EventMessage(
            parent_event_id=event.event_id,
            edge_id=e_out.edge_id,
            value=a,
            arrival_tick=t + e_out.delay_ticks
        )

    return event.event_id
```

### 10.2 Credit processing

```text
function APPLY_CREDIT(credit_id, feedback_tick T, root_packets):
    # pending[event_id] stores an accumulated scalar and minimum hop count.
    pending = AGGREGATE_BY_EVENT_ID(root_packets)
    processed = empty set

    while pending is not empty:
        # Greatest event tick first; event_id breaks ties deterministically.
        event_id = POP_LATEST_EVENT(pending)
        packet_sum, hop = pending[event_id]

        if event_id in processed:
            raise DuplicateProcessingError

        event = TRACE_STORE.get(event_id)
        if event is missing:
            LOG_STOP("missing_or_expired")
            continue

        c = clip(packet_sum, -Cmax, Cmax)
        age = T - event.tick

        if age < 0:
            raise CausalityError
        if age > H_trace:
            LOG_STOP("trace_horizon")
            continue
        if hop >= H_hop:
            LOG_STOP("hop_horizon")
            continue
        if abs(c) < c_min:
            LOG_STOP("small_credit")
            continue

        D = eps_route
        for trace r in event.edge_traces:
            D = D + abs(r.omission_effect)

        if D == eps_route:
            LOG_STOP("zero_omission")
            processed.add(event_id)
            continue

        proposals = []
        routes = []

        for trace r in event.edge_traces sorted by r.edge_id:
            q = abs(r.omission_effect) / D
            edge_credit = c * q

            raw_dw = (
                eta
                * (lambda_age ** age)
                * edge_credit
                * r.weight_secant
            )
            dw = clip(raw_dw, -Delta_max, Delta_max)
            proposals.append((r.edge_id, dw))

            parent_credit = (
                gamma_route
                * edge_credit
                * sign(r.source_secant)
            )
            routes.append((r.parent_event_id, parent_credit, hop + 1))

        # Traces and proposals were computed before any local mutation.
        for edge_id, dw in proposals sorted by edge_id:
            EDGE[edge_id].weight = clip(
                EDGE[edge_id].weight + dw,
                -Wmax,
                Wmax
            )

        processed.add(event_id)

        for parent_id, parent_credit, parent_hop in routes:
            # Threshold only the aggregate c when parent_id is processed.
            # Individually small reconvergent packets may have a useful sum.
            if parent_credit != 0:
                pending[parent_id].credit += parent_credit
                pending[parent_id].hop = min(
                    pending[parent_id].hop,
                    parent_hop
                )
```

An implementation may optimize storage and scheduling, but it must produce the same
records, update values, routing values, and deterministic order as this reference.

**Development clarification (2026-09-01, before confirmatory execution):** the
earlier pseudocode thresholded each routed packet before reconvergent packets
could be summed. That contradicted Sections 7.1 and 7.5 and validation test 11,
all of which define `c_min` on the aggregated event credit. The pseudocode and
simulator now apply the threshold only after aggregation. The pre-correction
development result remains recorded; no confirmatory seed was used.

---

## 11. Strict locality and access contract

This contract is more important than the class layout. A run counts as CCF-v0 only
if every item below is obeyed.

### 11.1 Forward-time access

During one unit event, unit \(j\) may read:

- its own fixed bias and last-emitted value;
- messages that arrived at \(j\) at the current tick;
- weights and delays of its incident incoming/outgoing edges;
- the activation function and frozen hyperparameters;
- parent event identifiers carried by the arriving messages.

It may not read:

- the task target or terminal outcome;
- another hidden unit's private state;
- nonincident weights;
- a graph-wide activation array;
- future messages or future rewards.

### 11.2 Credit-time access

When credit reaches a `UnitEvent`, the local credit handler may read:

- the arriving packet and its own `UnitEvent`;
- the event's immutable `EdgeTrace` values;
- the current weights of the referenced incoming edges;
- the frozen CCF hyperparameters;
- the parent event identifiers inside those traces.

It may not read:

- a global loss tensor;
- a hidden-layer target created by the controller;
- gradients from autograd;
- Jacobians from downstream units;
- unrelated traces;
- examples from a replay buffer;
- the full graph to search for substitute routes;
- a graph-wide normalizer.

### 11.3 Controller exception

The experiment controller may inject inputs, query designated outputs, calculate the
root credit using a frozen adapter, and record metrics. It must not directly modify
hidden weights or hidden traces.

### 11.4 Audit requirement

The reference implementation must be runnable with automatic differentiation
libraries absent or disabled. A locality audit should expose a narrow method such as

```text
unit.receive(messages)
unit.receive_credit(unit_event, credit_packet)
```

and fail a test if those methods access the global model object.

---

## 12. Frozen default hyperparameters

These are engineering defaults for the first implementation. An experiment protocol
may replace them only before running the corresponding experiment and must record the
replacement. Changing them after inspecting test outcomes creates a new experimental
condition.

| Name | Symbol | Default | Meaning |
|---|---:|---:|---|
| numeric type | - | `float64` | reference arithmetic |
| activation | \(\phi\) | `tanh` | all non-input units |
| emission threshold | \(\theta_{emit}\) | \(10^{-3}\) | minimum activation change to emit |
| learning rate | \(\eta\) | \(0.01\) | scale of local updates |
| age decay | \(\lambda_{age}\) | \(0.97\) per tick | discounts old traces |
| route attenuation | \(\gamma_{route}\) | \(0.90\) per hop | attenuates upstream packets |
| maximum absolute credit | \(C_{max}\) | \(1.0\) | clips aggregated event credit |
| maximum update | \(\Delta_{max}\) | \(0.05\) | clips one causal participation update |
| maximum absolute weight | \(W_{max}\) | \(3.0\) | hard weight bound |
| trace horizon | \(H_{trace}\) | `32 ticks` | maximum record age |
| hop horizon | \(H_{hop}\) | `16` | maximum backward route length |
| weight epsilon | \(\epsilon_w\) | \(10^{-12}\) | division guard |
| message epsilon | \(\epsilon_x\) | \(10^{-12}\) | division/zero-message guard |
| route epsilon | \(\epsilon_{route}\) | \(10^{-12}\) | local denominator guard |
| minimum credit | \(c_{min}\) | \(10^{-8}\) | packet termination threshold |

The graph seed, graph dimensions, initialization distribution, input encoding, output
adapter, episode length, and task schedule are experiment parameters, not hidden CCF
defaults. They must be frozen in the relevant experiment document.

---

## 13. Required invariants and executable assertions

### 13.1 Structural invariants

1. **Fixed topology:** the ordered edge list and every delay remain unchanged.
2. **Positive delay:** \(d_e\ge1\) for every edge.
3. **Unique identity:** every `UnitEvent` identifier is unique within a run.
4. **Earlier parent:** every recorded parent event has a strictly smaller tick than
   its child event.

### 13.2 Locality invariants

5. **Participation only:** an edge can update for a feedback occurrence only if an
   `EdgeTrace` for that edge lies on a recorded ancestor route from a credited output
   event.
6. **No reconstruction:** an expired/missing trace ends the branch.
7. **No hidden target:** only root output events receive controller-generated credit.
8. **No replay:** an update never requires re-running a stored task example.
9. **Local denominator:** Equation (6) includes traces from exactly one unit event.

### 13.3 Numeric invariants

10. **Bounded weights:** \(|w_e|\le W_{max}\) after every update.
11. **Bounded update:** \(|\Delta w_e|\le\Delta_{max}\) for each participation.
12. **Bounded event credit:** \(|c_j|\le C_{max}\) after aggregation.
13. **Conservative split:** \(\sum_e q_e\le1\).
14. **Contractive route at one event:**

\[
\sum_e|c_{\operatorname{parent}(e)}|
\le
\gamma_{route}|c_j|.
\]

15. **Finite arithmetic:** any NaN or infinity invalidates the run.

### 13.4 Determinism invariants

16. With the same code version, configuration, graph, seed, task sequence, and
floating-point environment, event identifiers, traces, packets, updates, and metrics
must match exactly.
17. Message reduction and edge commits use ascending `edge_id`; credit processing
uses descending event tick and deterministic event-id tie breaking.

---

## 14. Minimum validation tests before a learning experiment

The simulator must pass these mechanism tests before task performance is interpreted.

1. **One-edge positive-credit test:** for a nonsaturated monotone unit and nonzero
   input, positive credit increases the replayed activation.
2. **One-edge negative-credit test:** the same setup with negative credit decreases
   the replayed activation.
3. **Omission arithmetic test:** stored \(o\), \(g\), and \(h\) match direct
   recomputation to a stated tolerance.
4. **Share conservation test:** all \(q_e\ge0\) and \(\sum q_e\le1\).
5. **Untraced-edge test:** a nonparticipating edge remains bit-for-bit unchanged.
6. **Wrong-event test:** credit to one event never updates a different event from
   the same unit.
7. **Expiry test:** credit after `H_trace` produces no update and logs expiry.
8. **Hop-limit test:** no route exceeds `H_hop`.
9. **Delay-decay test:** identical traces at different ages have update magnitudes in
   the expected \(\lambda_{age}^{\Delta t}\) ratio.
10. **Route-sign test:** a negative source secant flips routed credit; a positive one
    preserves it.
11. **Reconvergence test:** packets to one parent event aggregate before clipping and
    before that parent is processed.
12. **Recurrent-graph termination test:** structural cycles cannot create credit
    cycles because recorded parent timestamps strictly decrease.
13. **Clip test:** extreme credit cannot violate `Delta_max` or `Wmax`.
14. **Deterministic replay-of-run test:** two fresh runs with the same seed produce
    identical event and update logs. This means rerunning the experiment from the
    beginning, not training replay inside CCF.
15. **No-autograd test:** the complete mechanism suite passes with gradient recording
    disabled and without calling `backward`, `grad`, or an equivalent API.

---

## 15. Compute and memory accounting

CCF-v0 is sparse, but its counterfactual calculations are not free.

For an emitted event with \(k\) arrived nonzero messages:

- the forward rule performs one actual activation evaluation;
- omission tracing performs \(k\) additional activation evaluations;
- the event stores \(k\) `EdgeTrace` records;
- local allocation and edge updates are \(O(k)\).

For one root feedback event, credit work is proportional to the number of retained
ancestor traces reached within the age and hop limits, not automatically to all
edges in the structural graph.

Every experiment must report at least:

- actual activation evaluations;
- omission activation evaluations;
- emitted event count;
- delivered message count;
- retained trace count and peak trace bytes;
- processed credit events and edge shares;
- weight-update count;
- expired, too-small, and hop-limited packet counts;
- wall-clock time on identified hardware.

Calling CCF efficient while excluding omission evaluations or trace memory would be
a reporting error.

---

## 16. Failure and falsification conditions

### 16.1 Immediate implementation failure

A run is invalid, not merely low-performing, if any of these occur:

- a required invariant in Section 13 fails;
- NaN or infinity appears;
- a nonparticipating edge changes;
- a trace is recomputed from current weights during credit processing;
- hidden credit is supplied by the controller;
- packet routing uses current graph search instead of recorded parent IDs;
- the result cannot be reproduced from the saved seed and configuration;
- compute accounting omits local counterfactual evaluations.

### 16.2 Mechanism-level falsification

CCF-v0 should be rejected or revised if it:

- fails the one-edge direction tests;
- is unstable under its preregistered numeric range despite clipping;
- cannot learn the smallest delayed-credit task above chance across preregistered
  seeds;
- performs no better than shuffled-credit or zero-credit controls;
- loses the effect when parent event identifiers are correct but preserves it when
  they are randomly permuted;
- requires a trace horizon covering essentially the full episode and graph, removing
  the proposed locality/efficiency advantage;
- depends on target leakage, replay, or unreported tuning;
- shows an improvement that disappears when compute, parameter count, and update
  count are matched.

Exact task thresholds belong in the experiment preregistration, not in this mechanism
document.

### 16.3 Novelty falsification

Even if CCF-v0 learns, a novelty claim must be rejected if its equations and update
behavior are equivalent, under a simple change of notation or normalization, to an
existing published or patented method and it yields no distinct testable prediction.

A useful result and a novel result are different claims. The project may still learn
from a non-novel mechanism, but it must name the prior method correctly.

---

## 17. Novelty and confound ledger

This ledger must remain visible while interpreting results. It is a checklist for a
future literature and patent comparison, not a claim that the listed families are
identical to CCF-v0.

| Possible overlap or confound | Why it threatens interpretation | Required response |
|---|---|---|
| Reward-modulated Hebbian / three-factor learning | A local eligibility-like term multiplied by delayed modulation is a known pattern. | Compare equations term by term; test whether omission allocation creates behavior not explained by a standard three-factor baseline. |
| Eligibility traces and TD-style credit | Age-decayed local records can reproduce familiar eligibility ideas. | Use a compute-matched eligibility-trace baseline and report exact differences. |
| E-prop and other local recurrent credit methods | Local eligibility plus broadcast or routed learning signals may be closely related. | Compare information access, recursion, truncation, and update equations. |
| Truncated backpropagation through time | Reverse traversal of recorded event ancestry may approximate a sparse/truncated backward pass. | Derive both updates on a linear network; identify conditions under which they coincide or diverge. |
| Backpropagation on an event graph | Local secants and routed signs can approximate a chain-rule direction. | Never claim "not backprop" only because autograd is absent; compare mathematical behavior directly. |
| Layer-wise relevance propagation, DeepLIFT, and contribution propagation | Omission-based local attribution and normalized redistribution resemble contribution methods. | Compare conservation rule, signs, baselines, and whether attribution is being reused as a learning rule. |
| Finite-difference and perturbation learning | Equation (1) is a one-message counterfactual perturbation. | Count its compute and compare with local finite-difference/perturbation baselines. |
| Event-driven or spiking networks | Sparse events and decaying traces are established concepts. | Attribute event sparsity separately from the credit rule. |
| Dynamic sparse networks | Later growth/pruning could produce gains unrelated to CCF. | Keep topology fixed until CCF-v0 passes; introduce structural plasticity only as a separate factorial condition. |
| Output supervision strength | The root adapter may contain most of the useful task information. | Freeze one adapter across methods and report exactly what it knows. |
| Architecture or initialization | A favorable graph can look like a better learning rule. | Use identical graphs, initial weights, parameter budgets, and seeds across compatible methods. |
| Emission threshold | Thresholding changes both compute and the data seen by the rule. | Ablate and compute-match threshold settings. |
| Clipping and normalization | Stability may come from engineering safeguards rather than causal routing. | Ablate route allocation, age decay, clipping, and route attenuation one at a time. |
| Hyperparameter search | More tuning can create a false win. | Give baselines equal search budgets and separate development seeds from locked test seeds. |
| Trace memory | A large trace buffer may hide an unfair memory advantage. | Report peak bytes and match or explicitly plot the memory/performance tradeoff. |
| Multiple updates per example | More local update operations can imitate faster learning. | Report and, where possible, match update count and arithmetic operations. |

Before any novelty statement, the project must complete a documented search of
relevant papers and patents and add citations plus an equation-level comparison.
The provisional name **Causal Credit Flow** must not be treated as evidence that the
mechanism is new.

---

## 18. Ablations that isolate the proposed mechanism

At minimum, later controlled experiments should include:

1. **No learning:** all weights frozen.
2. **Zero credit:** traces exist, but root credit is zero.
3. **Shuffled parent IDs:** preserve signal magnitudes while breaking causal ancestry.
4. **Uniform allocation:** replace \(q_e\) with equal shares among active inputs.
5. **Random allocation:** random nonnegative shares with the same total mass.
6. **Unsigned routing:** remove `sign(h)`.
7. **No upstream routing:** set \(\gamma_{route}=0\).
8. **No age decay:** set \(\lambda_{age}=1\).
9. **Shorter trace horizons:** measure the memory/performance curve.
10. **No omission rule:** use a conventional local eligibility term on the same
    graph with the same root credit.
11. **Dense emission:** set the event threshold to zero and count the extra compute.
12. **Clip removal or wider clips:** only inside a numerically guarded diagnostic.

An improvement can be attributed to the CCF omission mechanism only if the relevant
ablations remove or substantially reduce it while other budgets remain comparable.

---

## 19. Configuration and change-control rule

Every run must save:

- the exact CCF version string (`CCF-v0`);
- the hash of this specification and the implementation commit;
- the complete graph manifest;
- all values in Section 12;
- all task and output-adapter parameters;
- random seeds and deterministic-runtime settings;
- raw event, trace, packet, update, stop-reason, and metric logs.

If an equation, access rule, default behavior, or interpretation in this document is
changed after an experiment is observed, the changed rule is not CCF-v0. It must
receive a new version, and the earlier negative or positive result must remain in the
record.

---

## 20. Simple mental model

CCF-v0 can be remembered as four local questions:

1. **What arrived?** The unit considers only messages that reached it now.
2. **What mattered?** It omits each message once and measures how its own output
   changes.
3. **Who gets the delayed credit?** Messages get shares proportional to those local
   omission effects.
4. **Where does credit go next?** Each share updates that edge and follows the exact
   stored parent-event identifier, until the short trace ends.

That is the entire candidate mechanism. Whether these local operations produce
useful continual learning, retention, and composition is not assumed; it must be
answered by controlled experiments.
