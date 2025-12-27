use mpi::datatype::{Equivalence, Partition, PartitionMut};
#[allow(dead_code)]
use num::Zero;
//use mpi::traits::*;
use mpi::environment::Universe;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};

pub struct CommIfx {
    _universe: Option<Universe>,
    pub comm: SimpleCommunicator,
    pub rank: i32,
    pub size: i32,
}

impl CommIfx {
    pub fn init() -> Self {
        let (comm, _universe) = match mpi::initialize() {
            Some(universe) => (universe.world(), Some(universe)), // First time init
            None => (SimpleCommunicator::world(), None), // Already initialized
        };
        CommIfx {
            rank: comm.rank(),
            size: comm.size(),
            _universe,
            comm,
        }
    }

    pub fn finalize(&self) -> i32 {
        unsafe { mpi::ffi::MPI_Finalize() }
    }

    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    pub fn collect_counts<T>(&self, cx: T) -> std::vec::Vec<T>
    where
        T: Equivalence + Clone + Zero,
    {
        let mut rvx = vec![Zero::zero(); self.size as usize];
        self.comm.all_gather_into(&cx, &mut rvx[..]);
        rvx
    }

    pub fn a2a_counts<T>(&self, cx: &[T]) -> std::vec::Vec<T>
    where
        T: Equivalence + Clone + Zero,
    {
        let mut rvx = vec![Zero::zero(); self.size as usize];
        self.comm.all_to_all_into(cx, &mut rvx[..]);
        rvx
    }

    pub fn all_to_all_slice<T>(
        &self,
        snd_buf: &[T],
        snd_counts: &[i32],
        snd_displs: &[i32],
    ) -> Vec<T>
    where
        T: Equivalence + Clone + Zero,
    {
        let px = Partition::new(snd_buf, snd_counts, snd_displs);
        let rcv_counts = self.a2a_counts(snd_counts);
        let rcv_displs = rcv_counts
            .iter()
            .scan(0i32, |state, x| {
                let cstate = *state;
                *state += x;
                Some(cstate)
            })
            .collect::<Vec<i32>>();
        let rcv_total = rcv_counts.iter().sum::<i32>() as usize;
        let mut rvx = vec![Zero::zero(); rcv_total];
        let mut rcv_px = PartitionMut::new(&mut rvx, rcv_counts, rcv_displs);
        self.comm.all_to_all_varcount_into(&px, &mut rcv_px);
        rvx
    }

    pub fn all_to_all_vec<T>(&self, snd_buf: &[T], snd_counts: &[i32]) -> Vec<T>
    where
        T: Equivalence + Clone + Zero,
    {
        let snd_displs = snd_counts
            .iter()
            .scan(0i32, |state, x| {
                let cstate = *state;
                *state += x;
                Some(cstate)
            })
            .collect::<Vec<i32>>();
        self.all_to_all_slice(snd_buf, snd_counts, &snd_displs)
    }
}
