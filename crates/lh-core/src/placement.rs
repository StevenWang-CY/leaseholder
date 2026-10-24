//! Placement: which GPU node a new session goes to, when one control plane fronts several engines.
//!
//! KV never moves between nodes, so a session stays on the node that holds its KV. The caller keeps each
//! session's node and asks for one only for a session that has none: a new session, one idle long enough to be
//! placed again, or one whose node stopped answering. [`Policy::place`] chooses from what each node holds at that
//! moment, as the caller reports it ([`Node`]). Like the admission policies, it is pure and has no clock.
//! - [`Policy::Dedicated`]: each node serves the classes it names, so calls and agents have GPUs of their own.
//! - [`Policy::LeastLoaded`]: the stock pool. A new session goes to the node with the fewest requests in flight.
//! - [`Policy::Leaseholder`]: Leaseholder's own. An agent session goes to the node with the least estimated
//!   background KV, and a call to the node where its turns will find headroom: the fewest calls, then the fewest
//!   armed for a turn now, then the most free KV.
//!
//! Every policy breaks its ties by the fewest requests in flight, and then by the earlier node in the caller's
//! order.

use std::cmp::Reverse;

use lh_proto::SloClass;

/// A placement policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// A session goes to a node that serves its class: of several, the one with the fewest requests in flight.
    Dedicated,
    /// A session goes to the node with the fewest requests in flight.
    LeastLoaded,
    /// An agent session (background or interactive) goes to the node with the least estimated background KV. A
    /// realtime session goes to the node with the fewest realtime sessions, then the fewest of them armed, then
    /// the most free KV: the headroom its turns will need.
    Leaseholder,
}

/// A node as a placement sees it: what it serves, and what it holds now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Node {
    /// Whether it answers its readiness probe. A node that does not takes no new session.
    pub ready: bool,
    /// Whether it serves each class, strictest first (policy dedicated).
    pub serves: [bool; 3],
    /// Its requests in flight.
    pub in_flight: u32,
    /// The realtime sessions placed on it (policy leaseholder).
    pub realtime_sessions: u32,
    /// Those of its realtime sessions armed for a turn now, as its headroom's trigger arms them (policy
    /// leaseholder).
    pub armed: u32,
    /// The estimated tokens of the background requests it holds, in flight and waiting (policy leaseholder).
    pub background_tokens: u64,
    /// Its KV pool less the estimated tokens of its background requests in flight (policy leaseholder).
    pub free_tokens: u64,
}

impl Policy {
    pub const ALL: [Self; 3] = [Self::Dedicated, Self::LeastLoaded, Self::Leaseholder];

    /// The name, also the configuration's value and a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dedicated => "dedicated",
            Self::LeastLoaded => "least-loaded",
            Self::Leaseholder => "leaseholder",
        }
    }

    /// The node a new session of `class` goes to, as an index into `nodes`; `None` when no ready node takes it.
    pub fn place(self, class: SloClass, nodes: &[Node]) -> Option<usize> {
        let ready = nodes.iter().enumerate().filter(|(_, node)| node.ready);
        let chosen = match self {
            Self::Dedicated => ready
                .filter(|(_, node)| node.serves[class as usize])
                .min_by_key(|(_, node)| node.in_flight),
            Self::LeastLoaded => ready.min_by_key(|(_, node)| node.in_flight),
            Self::Leaseholder if class == SloClass::Realtime => ready.min_by_key(|(_, node)| {
                let headroom = (node.armed, Reverse(node.free_tokens));
                (node.realtime_sessions, headroom, node.in_flight)
            }),
            Self::Leaseholder => {
                ready.min_by_key(|(_, node)| (node.background_tokens, node.in_flight))
            }
        };
        chosen.map(|(index, _)| index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SloClass::*;

    /// A ready node that serves every class and holds nothing.
    fn idle() -> Node {
        Node {
            ready: true,
            serves: [true; 3],
            ..Node::default()
        }
    }

    #[test]
    fn dedicated_places_each_class_on_a_node_that_serves_it() {
        let voice = Node {
            serves: [true, false, false],
            in_flight: 40,
            ..idle()
        };
        let agents = Node {
            serves: [false, true, true],
            in_flight: 9,
            ..idle()
        };
        let place = |class, nodes: &[Node]| Policy::Dedicated.place(class, nodes);
        let nodes = [voice, agents];
        assert_eq!(place(Realtime, &nodes), Some(0), "however busy");
        assert_eq!(place(Background, &nodes), Some(1));
        assert_eq!(place(Interactive, &nodes), Some(1));
        // Of the nodes that serve a class, the one with the fewest requests in flight.
        let quieter = Node {
            in_flight: 3,
            ..agents
        };
        assert_eq!(place(Background, &[voice, agents, quieter]), Some(2));
        assert_eq!(place(Background, &[voice, voice]), None);
    }

    #[test]
    fn least_loaded_takes_the_node_with_the_fewest_requests_in_flight() {
        let busy = Node {
            in_flight: 5,
            ..idle()
        };
        let quiet = Node {
            in_flight: 2,
            ..idle()
        };
        for class in SloClass::ALL {
            assert_eq!(Policy::LeastLoaded.place(class, &[busy, quiet]), Some(1));
        }
        // A tie goes to the earlier node, and what a node serves does not matter.
        let voice_only = Node {
            serves: [true, false, false],
            ..quiet
        };
        assert_eq!(
            Policy::LeastLoaded.place(Background, &[voice_only, quiet]),
            Some(0)
        );
    }

    #[test]
    fn leaseholder_places_agents_by_their_kv() {
        let full = Node {
            background_tokens: 30_000,
            in_flight: 1,
            ..idle()
        };
        let roomy = Node {
            background_tokens: 12_000,
            in_flight: 9,
            ..idle()
        };
        for class in [Background, Interactive] {
            assert_eq!(Policy::Leaseholder.place(class, &[full, roomy]), Some(1));
        }
        // With equal KV, as at the start, requests in flight spread sessions that arrive together.
        let empty = idle();
        let started = Node {
            in_flight: 1,
            ..idle()
        };
        assert_eq!(
            Policy::Leaseholder.place(Background, &[started, empty]),
            Some(1)
        );
    }

    #[test]
    fn leaseholder_places_a_call_where_its_turns_find_headroom() {
        let node = |realtime_sessions, armed, free_tokens| Node {
            realtime_sessions,
            armed,
            free_tokens,
            // The calls' rules come first: agents' requests in flight only break a tie.
            in_flight: 20 - realtime_sessions,
            ..idle()
        };
        let place = |nodes: &[Node]| Policy::Leaseholder.place(Realtime, nodes);
        // The fewest calls, however their turns stand now and whatever KV is free.
        assert_eq!(place(&[node(2, 0, 36_000), node(1, 1, 2_000)]), Some(1));
        // Then the fewest armed for a turn now.
        assert_eq!(place(&[node(1, 1, 36_000), node(1, 0, 2_000)]), Some(1));
        // Then the most free KV.
        assert_eq!(place(&[node(1, 0, 20_000), node(1, 0, 30_000)]), Some(1));
        assert_eq!(place(&[node(0, 0, 9_000), node(0, 0, 9_000)]), Some(0));
    }

    #[test]
    fn a_node_that_is_not_ready_takes_no_session() {
        let down = Node {
            ready: false,
            ..idle()
        };
        let busy = Node {
            in_flight: 50,
            realtime_sessions: 3,
            background_tokens: 36_000,
            ..idle()
        };
        for policy in Policy::ALL {
            for class in SloClass::ALL {
                assert_eq!(policy.place(class, &[down, busy]), Some(1), "{policy:?}");
                assert_eq!(policy.place(class, &[down, down]), None, "{policy:?}");
            }
            assert_eq!(policy.place(Realtime, &[]), None);
        }
    }
}
