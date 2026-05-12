use mpi::datatype::Equivalence;
use mpi::topology::SimpleCommunicator;
use num::Zero;
use sope::collective::allgather_one;
use sope::comm::WorldComm;

/// Light wrapper around `sope::comm::WorldComm`.
///
/// # Description
/// Caches the world `comm`, the local `rank`, and the world `size` so
/// that callers do not have to repeatedly query them. Process `0` is
/// treated as the conventional root.
pub struct CommIfx {
    world: WorldComm,
    pub rank: i32,
    pub size: i32,
}

impl CommIfx {
    /// Initialize WorldComm from sope library
    pub fn init() -> Self {
        let world = WorldComm::init();
        CommIfx {
            rank: world.rank,
            size: world.size,
            world,
        }
    }

    /// Explicitly call `MPI_Finalize`.
    pub fn finalize(&self) -> i32 {
        self.world.finalize()
    }

    /// Returns `true` if this process is the root (rank `0`).
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Returns a reference to SimpleCommunicator
    pub fn comm(&self) -> &SimpleCommunicator {
        &self.world.comm
    }

    /// Collect T values from all the processes
    pub fn collect_counts<T>(&self, cx: T) -> std::vec::Vec<T>
    where
        T: Equivalence + Clone + Zero + Default,
    {
        allgather_one(&cx, self.comm())
            .unwrap_or(vec![T::default(); self.size as usize])
    }
}
