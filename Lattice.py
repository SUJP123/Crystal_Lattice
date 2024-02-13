import itertools
import math
import hoomd
import numpy as np
import gsd.hoomd
import hoomd.md

def run(fname, fname2):
    m = 4
    N_particles = 4 * m**3

    spacing = 1.3
    K = math.ceil(N_particles ** (1 / 3))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = position[0:N_particles]
    frame.particles.typeid = [0] * N_particles
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A']

    with gsd.hoomd.open(name=fname, mode='x') as f:
        f.append(frame)

    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    simulation.create_state_from_gsd(filename=fname)

    integrator = hoomd.md.Integrator(dt=0.005)
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2.5
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
    integrator.methods.append(nvt)

    simulation.operations.integrator = integrator
    snapshot = simulation.state.get_snapshot()
    snapshot.particles.velocity[0:5]
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
    
    simulation.run(10000)

    hoomd.write.GSD.write(state=simulation.state, filename=fname2, mode='xb')


while __name__ == "__main__":
    print(" ")
    print("Name current file (notate with .gsd)")
    name = input()
    print(" ")
    print("Name another file (notate with .gsd)")
    name2 = input()
    run(name, name2)
