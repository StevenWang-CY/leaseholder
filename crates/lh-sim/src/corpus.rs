//! A trace source with replay plans built on first use.

use std::path::Path;
use std::rc::Rc;

use crate::aiperf::{PlanOptions, ReplayPlan};
use crate::trace::TraceSource;

/// Traces in AIPerf's sampler order, each with its replay plan, built on first use.
#[derive(Debug)]
pub struct Corpus {
    source: TraceSource,
    opts: PlanOptions,
    plans: Vec<Option<Rc<ReplayPlan>>>,
}

impl Corpus {
    /// Opens a directory of `<trace id>.json` or a JSONL file; plans are built with `opts`.
    pub fn open(path: &Path, opts: PlanOptions) -> Result<Self, String> {
        let source = TraceSource::open(path).map_err(|e| e.to_string())?;
        if source.is_empty() {
            return Err(format!("{}: no traces", path.display()));
        }
        Ok(Self::new(source, opts))
    }

    /// A corpus over an opened source, which may be empty.
    pub fn new(source: TraceSource, opts: PlanOptions) -> Self {
        let plans = vec![None; source.len()];
        Self {
            source,
            opts,
            plans,
        }
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// How the plans are built.
    pub fn options(&self) -> &PlanOptions {
        &self.opts
    }

    /// The plan of trace `index`, built from the source the first time it is needed.
    pub fn plan(&mut self, index: usize) -> Result<Rc<ReplayPlan>, String> {
        if let Some(plan) = &self.plans[index] {
            return Ok(plan.clone());
        }
        let trace = self.source.load(index).map_err(|e| e.to_string())?;
        let plan = Rc::new(ReplayPlan::build(trace, &self.opts)?);
        self.plans[index] = Some(plan.clone());
        Ok(plan)
    }

    /// The index of the trace with id `id`.
    pub fn find(&mut self, id: &str) -> Result<Option<usize>, String> {
        if let Some(i) = self.source.file_index(id)
            && self.plan(i)?.trace_id == id
        {
            return Ok(Some(i));
        }
        for i in 0..self.len() {
            if self.plan(i)?.trace_id == id {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }
}
