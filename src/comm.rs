use mpi::datatype::Equivalence;
use mpi::topology::SimpleCommunicator;
use num::Zero;
use sope::collective::allgather_one;
use sope::comm::WorldComm;

pub struct CommIfx {
    world: WorldComm,
    pub rank: i32,
    pub size: i32,
}

impl CommIfx {
    pub fn init() -> Self {
        let world = WorldComm::init();
        CommIfx {
            rank: world.rank,
            size: world.size,
            world,
        }
    }

    pub fn finalize(&self) -> i32 {
        self.world.finalize()
    }

    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    pub fn comm(&self) -> &SimpleCommunicator {
        &self.world.comm
    }

    pub fn collect_counts<T>(&self, cx: T) -> std::vec::Vec<T>
    where
        T: Equivalence + Clone + Zero + Default,
    {
        allgather_one(&cx, self.comm())
            .unwrap_or(vec![T::default(); self.size as usize])
    }
}
