###cython: profile=True
##cython: boundscheck=False
##cython: wraparound=False
##cython: nonecheck=False
##cython: initializedcheck=False
#cython: cdivision=True
from __future__ import division, print_function
cimport cython
import numpy as np
cimport numpy as np

#mport numpy as np
from cpython cimport bool
from heapq import heappush, heappop

from libc.stdlib cimport rand, srand, RAND_MAX
srand(0)
np.random.seed()

# generates N random numbers between 0 and N_MAX
cdef inline np.int64_t rand_int(int N_MAX):
    return np.random.randint(N_MAX)
    #return rand() % N_MAX

# Defining simple python functions that will be called often on a "C-level"
cdef inline np.float64_t rand_exp(np.float64_t mean):
    return np.random.exponential(mean) if mean > 0 else 0

cdef inline float rand_float():
    #return np.random.random()
    return rand() / float(RAND_MAX)

cdef inline int64sign(np.int64_t x):
    if x > 0:
        return +1
    else:
        return 0

cdef inline int64abs(np.int64_t x):
    if x > 0:
        return x
    else:
        return -x

cdef inline int64not(np.int64_t x):
    if x == 0:
        return 1
    else:
        return 0

cdef class System:
    # Declaring variables 
    cdef np.int64_t L           # Lattice Size(?)
    cdef np.int64_t N           # Number of Loci
    cdef np.float64_t time      # Walk time(?)

    # vels have 4*N elements
    # 0:N       - outer steps (<-) for left legs 
    # N:2N      - inner steps (<-) for right legs 
    # 2N:3N     - inner steps (->) for left legs 
    # 3N:4N     - outer steps (->) for right legs
    cdef np.float_t [:] vels

    cdef np.float_t [:] legswitchrates 
    cdef np.int64_t [:] leading_legs

    cdef np.float_t [:] lifespans
    cdef np.float_t [:] rebinding_times

    cdef np.int64_t [:] locs

    cdef np.float_t [:] perms
    cdef np.int64_t [:] lattice

    def __cinit__(self, 
            L,
            N,
            vels,
            legswitchrates,
            lifespans, 
            rebinding_times,
            init_locs=None,
            perms=None):
        self.L = L
        self.N = N  # Number of loop extruders
        self.lattice = -1 * np.ones(L, dtype=np.int64)  # Number of Locii
        self.locs = -1 * np.ones(2*N, dtype=np.int64)   # The 2N legs positions!
        self.vels = vels

        self.legswitchrates = legswitchrates
        self.leading_legs = -1 * np.ones(N, dtype=np.int64)

        self.lifespans = lifespans
        self.rebinding_times = rebinding_times

        if perms is None:
            self.perms = np.ones(L+1, dtype=np.float64)
        else:
            self.perms = perms

        self.perms[0] = 0.0
        self.perms[-1] = 0.0

        cdef np.int64_t i

        # Initialize non-random loops
        for i in range(2 * self.N):
            # Check if the loop is preinitialized.
            if (init_locs[i] < 0):
                continue

            # Populate a site.
            self.locs[i] = init_locs[i]
            self.lattice[self.locs[i]] = i
            if i < self.N:
                self.leading_legs[self.locs[i]] = rand_int(2)

    cdef np.int64_t make_step(System self, np.int64_t leg_idx, np.int64_t direction):
        """
        The variable `direction` can only take values +1 or -1.
        """

        cdef np.int64_t new_pos = self.locs[leg_idx] + direction
        return self.move_leg(leg_idx, new_pos)

    cdef np.int64_t move_leg(System self, np.int64_t leg_idx, np.int64_t new_pos):
        if (new_pos >= 0) and (self.lattice[new_pos] >=0):
            return 0

        cdef np.int64_t prev_pos

        prev_pos = self.locs[leg_idx]
        self.locs[leg_idx] = new_pos

        if prev_pos >= 0:
            if self.lattice[prev_pos] < 0:
                return 0
            self.lattice[prev_pos] = -1

        if new_pos >= 0:
            self.lattice[new_pos] = leg_idx

        return 1

    cdef np.int64_t switch_leading_leg(System self, np.int64_t leg_idx, np.int64_t new_leading_leg):
        self.leading_legs[leg_idx] = new_leading_leg 

        return 1

    cdef np.int64_t check_system(System self):
        okay = 1
        cdef np.int64_t i
        for i in range(self.N):
            if (self.locs[i] == self.locs[i+self.N]):
                print('loop ' , i, 'has both legs at ', self.locs[i])
                okay = 0
            if (self.locs[i] >= self.L):
                print('leg ', i, 'is located outside of the system: ', self.locs[i])
                okay = 0
            if (self.locs[i+self.N] >= self.L):
                print('leg ', i+self.N, 'is located outside of the system: ', self.locs[i+self.N])
                okay = 0
            if (((self.locs[i] < 0) and (self.locs[i+self.N] >= 0 ))
                or ((self.locs[i] >= 0) and (self.locs[i+self.N] < 0 ))):
                print('the legs of the loop', i, 'are inconsistent: ', self.locs[i], self.locs[i+self.N])
                okay = 0
        return okay

cdef class Event_t:
    cdef public np.float64_t time
    cdef public np.int64_t event_idx

    def __cinit__(Event_t self, np.float_t time, np.int64_t event_idx):
        self.time = time
        self.event_idx = event_idx

    def __richcmp__(Event_t self, Event_t other, int op):
        if op == 0:
            return 1 if self.time <  other.time else 0
        elif op == 1:
            return 1 if self.time <= other.time else 0
        elif op == 2:
            return 1 if self.time == other.time else 0
        elif op == 3:
            return 1 if self.time != other.time else 0
        elif op == 4:
            return 1 if self.time >  other.time else 0
        elif op == 5:
            return 1 if self.time >= other.time else 0


cdef class Event_heap:
    """Taken from the official Python website"""
    cdef public list heap
    cdef public dict entry_finder

    def __cinit__(self):
        self.heap = list()
        self.entry_finder = dict()

    cdef add_event(Event_heap self, np.int64_t event_idx, np.float64_t time=0):
        'Add a new event or update the time of an existing event.'
        if event_idx in self.entry_finder:
            self.remove_event(event_idx)
        cdef Event_t entry = Event_t(time, event_idx)
        self.entry_finder[event_idx] = entry
        heappush(self.heap, entry)

    cdef remove_event(Event_heap self, np.int64_t event_idx):
        'Mark an existing event as REMOVED.'
        cdef Event_t entry
        if event_idx in self.entry_finder:
            entry = self.entry_finder.pop(event_idx)
            entry.event_idx = -1

    cdef Event_t pop_event(Event_heap self):
        'Remove and return the closest event.'
        cdef Event_t entry
        while self.heap:
            entry = heappop(self.heap)
            if entry.event_idx != -1:
                del self.entry_finder[entry.event_idx]
                return entry
        return Event_t(0, 0.0)


cdef regenerate_event(System system, Event_heap evheap, np.int64_t event_idx):
    """
    Regenerate an event in an event heap. If the event is currently impossible (e.g. a step
    onto an occupied site) then the new event is not created, but the existing event is not
    modified.

    Possible events:
    0 to 2N-1   : a step to the left
    2N to 4N-1  : a step to the right
    4N to 5N-1  : switch leading leg 
    5N to 6N-1  : passive unbinding
    6N to 7N-1  : rebinding to a randomly chosen site
    """

    cdef np.int64_t leg_idx, loop_idx
    cdef np.int64_t direction
    cdef np.float_t local_vel

    # A step to the left or to the right.
    if (event_idx < 4 * system.N) :
        if event_idx < 2 * system.N:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N
            direction = 1

        if (system.locs[leg_idx] >= 0):
            # Local velocity = velocity * permeability
            local_vel = (
                system.perms[system.locs[leg_idx] + (direction+1)//2]
                * system.vels[leg_idx + (direction+1) * system.N]
            )
    
            if local_vel > 0:
                if system.lattice[system.locs[leg_idx] + direction] < 0:
                    evheap.add_event(
                        event_idx,
                        system.time + rand_exp(1.0 / local_vel))

    # Switch leading leg.
    elif (event_idx >= 4 * system.N) and (event_idx < 5 * system.N):
        loop_idx = event_idx - 4 * system.N
        if (system.locs[loop_idx] >= 0) and (system.locs[loop_idx+system.N] >= 0):
            evheap.add_event(
                event_idx,
                system.time + rand_exp(1.0 / system.legswitchrates[loop_idx]))
        else:
            evheap.remove_event(event_idx)

    # Passive unbinding.
    elif (event_idx >= 5 * system.N) and (event_idx < 6 * system.N):
        loop_idx = event_idx - 5 * system.N
        if (system.locs[loop_idx] >= 0) and (system.locs[loop_idx+system.N] >= 0):
            evheap.add_event(
                event_idx,
                system.time + rand_exp(system.lifespans[loop_idx]))

    # Rebinding from the solution to a random site.
    elif (event_idx >= 6 * system.N) and (event_idx < 7 * system.N):
        loop_idx = event_idx - 6 * system.N
        if (system.locs[loop_idx] < 0) and (system.locs[loop_idx+system.N] < 0):
            evheap.add_event(
                event_idx,
                system.time + rand_exp(system.rebinding_times[loop_idx]))

cdef regenerate_neighbours(System system, Event_heap evheap, np.int64_t pos):
    """
    Regenerate the motion events for the adjacent loop legs.
    Use to unblock the previous neighbors and block the new ones.
    """
    # regenerate left step for the neighbour on the right
    if (pos<system.L-1) and system.lattice[pos+1] >= 0:
        regenerate_event(system, evheap, system.lattice[pos+1])

    # regenerate right step for the neighbour on the left
    if (pos>0) and system.lattice[pos-1] >= 0:
        regenerate_event(system, evheap, system.lattice[pos-1] + 2 * system.N)
 
cdef regenerate_all_loop_events(
    System system, Event_heap evheap, np.int64_t loop_idx):
    """
    Regenerate all possible events for a loop. Includes the four possible motions
    leading leg switching, passive unbinding and rebinding.
    """
    # Note, loop_idx is between [0,...,N-1]!!!
    regenerate_event(system, evheap, loop_idx)
    regenerate_event(system, evheap, loop_idx + system.N)
    regenerate_event(system, evheap, loop_idx + 2 * system.N)
    regenerate_event(system, evheap, loop_idx + 3 * system.N)
    regenerate_event(system, evheap, loop_idx + 4 * system.N)
    regenerate_event(system, evheap, loop_idx + 5 * system.N)
    regenerate_event(system, evheap, loop_idx + 6 * system.N)


cdef np.int64_t do_event(System system, Event_heap evheap, np.int64_t event_idx) except 0:
    """
    Apply an event from a heap on the system and then regenerate it.
    If the event is currently impossible (e.g. a step onto an occupied site),
    it is not applied, however, no warning is raised.

    Also, partially checks the system for consistency. Returns 0 is the system
    is not consistent (a very bad sign), otherwise returns 1 if the event was a step
    and 3 if the event was rebinding.

    Possible events:
    0 to 2N-1 : a step to the left
    2N to 4N-1 : a step to the right
    4N to 5N-1 : switch leading leg 
    5N to 6N-1 : passive unbinding
    6N to 7N-1 : rebinding to a randomly chosen site
    """

    cdef np.int64_t status
    cdef np.int64_t new_pos, leg_idx, prev_pos, direction, loop_idx

    if event_idx < 4 * system.N:
        # Take a step
        if event_idx < 2 * system.N:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N
            direction = 1

        loop_idx = leg_idx if leg_idx < system.N else leg_idx - system.N

        prev_pos = system.locs[leg_idx]
        # check if the loop was attached to the chromatin
        status = 1
        if (prev_pos >= 0):
            # make a step only if
            # ...there is no boundary 
            # ...the new position is unoccupied
            # ...an outer step is made by the leading leg
            if (
                (system.perms[prev_pos + (direction + 1) // 2] > 0)
                and (system.lattice[prev_pos+direction] < 0)
                and (
                        # backstepping
                        (
                        (event_idx >= system.N) 
                        and
                        (event_idx < 3 * system.N) 
                        ) 
                    or
                    (
                        # forward stepping with the leading leg
                        ((system.leading_legs[loop_idx] == 0)
                            and (event_idx < system.N))
                        or
                        ((system.leading_legs[loop_idx] == 1)
                            and (event_idx >= 3 * system.N))
                    )
                    )
            ):
                    status *= system.make_step(leg_idx, direction)
                    # regenerate events for the previous and the new neighbors
                    regenerate_neighbours(system, evheap, prev_pos)
                    regenerate_neighbours(system, evheap, prev_pos+direction)

            # regenerate the performed event
            regenerate_event(system, evheap, event_idx)

    elif (event_idx >= 4 * system.N) and (event_idx < 5 * system.N):
        # switch the leading leg
        loop_idx = event_idx - 4 * system.N
        status = 2
        # check if the loop was attached to the chromatin
        if (system.locs[loop_idx] < 0) or (system.locs[loop_idx+system.N] < 0):
            #print('leading leg status 0')
            status = 0
        
        system.switch_leading_leg(loop_idx, 1 - system.leading_legs[loop_idx])

        # regenerate the performed event
        regenerate_event(system, evheap, event_idx)

    elif (event_idx >= 5 * system.N) and (event_idx < 6 * system.N):
        # rebind the loop
        loop_idx = event_idx - 5 * system.N

        status = 2
        # check if the loop was attached to the chromatin
        if (system.locs[loop_idx] < 0) or (system.locs[loop_idx+system.N] < 0):
            #print('unbinding status 0')
            status = 0

        # save previous positions, but don't update neighbours until the loop
        # has moved
        prev_pos1 = system.locs[loop_idx]
        prev_pos2 = system.locs[loop_idx + system.N]

        status *= system.move_leg(loop_idx, -1)
        status *= system.move_leg(loop_idx+system.N, -1)
        status *= system.switch_leading_leg(loop_idx, -1)

        # regenerate events for the loop itself and for its previous neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)

        # update the neighbours after the loop has moved
        regenerate_neighbours(system, evheap, prev_pos1)
        regenerate_neighbours(system, evheap, prev_pos2)

    elif (event_idx >= 6 * system.N) and (event_idx < 7 * system.N):
        loop_idx = event_idx - 6 * system.N

        status = 2
        # check if the loop was not attached to the chromatin
        if (
            (system.locs[loop_idx] >= 0) 
            or (system.locs[loop_idx+system.N] >= 0)
            or (system.leading_legs[loop_idx] >= 0)
            ):
            #print('rebinding status 0')
            status = 0

        # find a new position for the LEF (a brute force method, can be
        # improved)
        while True:
            new_pos = rand_int(system.L-1)
            if (system.lattice[new_pos] < 0
                and system.lattice[new_pos+1] < 0
                and system.perms[new_pos+1] > 0):
                break
            new_pos = rand_int(system.L-1)

        # rebind the loop
        status *= system.move_leg(loop_idx, new_pos)
        status *= system.move_leg(loop_idx+system.N, new_pos+1)
        status *= system.switch_leading_leg(loop_idx, rand_int(2))

        # regenerate events for the loop itself and for its new neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)
        regenerate_neighbours(system, evheap, new_pos)
        regenerate_neighbours(system, evheap, new_pos + 1)
    else:
        print('event_idx assumed a forbidden value :', event_idx)

    return status


cpdef simulate(p, verbose=True):
    '''Simulate a system of loop extruding LEFs on a 1d lattice.
    Allows to simulate two different types of LEFs, with different
    residence times and rates of backstep.

    Parameters
    ----------
    p : a dictionary with parameters
        PROCESS_NAME : the title of the simulation
        L : the number of sites in the lattice
        N : the number of LEFs
        R_EXTEND : the rate of loop extension,
            can be set globally with a float,
            or individually with an array of floats
        R_SHRINK : the rate of LEF backsteps,
            can be set globally with a float,
            or individually with an array of floats
        R_SWITCH : the rate with which a LEF switches its leading leg,
            can be set globally with a float,
            or individually with an array of floats
        R_OFF : the rate of detaching from the polymer,
            can be set globally with a float,
            or individually with an array of floats
        REBINDING_TIME : the average time that a LEF spends in solution before
            rebinding to the polymer; can be set globally with a float,
            or individually with an array of floats
        INIT_L_SITES : the initial positions of the left legs of the LEFs,
                       If -1, the position of the LEF is chosen randomly,
                       with both legs next to each other. By default is -1 for
                       all LEFs.
        INIT_R_SITES : the initial positions of the right legs of the LEFs
        ACTIVATION_TIMES : the times at which the LEFs enter the system.
            By default equals 0 for all LEFs.
            Must be 0 for the LEFs with defined INIT_L_SITES
            and INIT_R_SITES.

        T_MAX : the duration of the simulation
        N_SNAPSHOTS : the number of time frames saved in the output. The frames
                      are evenly distributed between 0 and T_MAX.

    '''
    cdef char* PROCESS_NAME = p['PROCESS_NAME']

    cdef np.int64_t L = p['L']
    cdef np.int64_t N = np.round(p['N'])
    cdef np.float64_t T_MAX = p['T_MAX']
    cdef np.int64_t N_SNAPSHOTS = p['N_SNAPSHOTS']

    cdef np.int64_t i

    cdef np.float64_t [:] VELS = np.zeros(4*N, dtype=np.float64)
    cdef np.float64_t [:] LIFESPANS = np.zeros(N, dtype=np.float64)
    cdef np.float64_t [:] LEGSWITCHRATES = np.zeros(N, dtype=np.float64)
    cdef np.float64_t [:] REBINDING_TIMES = np.zeros(N, dtype=np.float64)

    for i in range(N):
        VELS[i]   = VELS[i+3*N] = p['R_EXTEND'][i] if type(p['R_EXTEND']) in (list, np.ndarray) else p['R_EXTEND']
        VELS[i+N] = VELS[i+2*N] = p['R_SHRINK'][i] if type(p['R_SHRINK']) in (list, np.ndarray) else p['R_SHRINK']
        LEGSWITCHRATES[i] = p['R_SWITCH'][i] if type(p['R_SWITCH']) in (list, np.ndarray) else p['R_SWITCH']
        LIFESPANS[i] = (1.0 / p['R_OFF'][i]) if type(p['R_OFF']) in (list, np.ndarray) else 1.0 / p['R_OFF']
        REBINDING_TIMES[i] = (
            (p['REBINDING_TIME'][i])
            if type(p.get('REBINDING_TIME',0)) in (list, np.ndarray)
            else p.get('REBINDING_TIME',0))

    cdef np.int64_t [:] INIT_LOCS = (-1) * np.ones(2*N, dtype=np.int64)
    if ('INIT_L_SITES' in p) and ('INIT_R_SITES' in p):
        for i in range(N):
            INIT_LOCS[i] = p['INIT_L_SITES'][i]
            INIT_LOCS[i+N] = p['INIT_R_SITES'][i]

    cdef np.float64_t [:] ACTIVATION_TIMES = p.get('ACTIVATION_TIMES',
        np.zeros(N, dtype=np.float64))

    for i in range(N):
        if INIT_LOCS[i] != -1:
            assert (INIT_LOCS[i+N] != -1)
            assert ACTIVATION_TIMES[i] == 0
        else:
            assert (INIT_LOCS[i+N] == -1)

    cdef np.float_t [:] PERMS = p.get('PERMS', None)
    if (not (PERMS is None)) and (PERMS.size != L+1):
        raise Exception(
            'The length of the provided array of permeabilities should be L+1')

    cdef System system = System(
        L, N, VELS, LEGSWITCHRATES, 
        LIFESPANS, REBINDING_TIMES,
        INIT_LOCS, PERMS)

    cdef np.int64_t [:,:] l_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.int64_t [:,:] r_sites_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.int64_t [:,:] leading_legs_traj = np.zeros((N_SNAPSHOTS, N), dtype=np.int64)
    cdef np.float64_t [:] ts_traj = np.zeros(N_SNAPSHOTS, dtype=np.float64)

    cdef np.int64_t last_event = 0

    cdef np.float64_t prev_snapshot_t = 0
    cdef np.float64_t tot_rate = 0
    cdef np.int64_t snapshot_idx = 0

    cdef Event_heap evheap = Event_heap()

    # Move LEFs onto the lattice at the corresponding activations times.
    # If the positions were predefined, initialize the fall-off time in the
    # standard way.
    for i in range(system.N):
        # if the loop location is not predefined, activate it
        # at the predetermined time
        if (INIT_LOCS[i] == -1) and (INIT_LOCS[i] == -1):
            evheap.add_event(i + 6 * system.N, ACTIVATION_TIMES[i])

        # otherwise, the loop is already placed on the lattice and we need to
        # regenerate all of its events and the motion of its neighbours
        else:
            regenerate_all_loop_events(system, evheap, i)
            regenerate_neighbours(system, evheap, INIT_LOCS[i])
            regenerate_neighbours(system, evheap, INIT_LOCS[i+system.N])

    cdef Event_t event
    cdef np.int64_t event_idx

    while snapshot_idx < N_SNAPSHOTS:
        event = evheap.pop_event()
        system.time = event.time
        event_idx = event.event_idx

        status = do_event(system, evheap, event_idx)

        if status == 0:
            print('an assertion failed somewhere')
            return 0

        if system.time > prev_snapshot_t + T_MAX / N_SNAPSHOTS:
            prev_snapshot_t = system.time
            l_sites_traj[snapshot_idx] = system.locs[:N]
            r_sites_traj[snapshot_idx] = system.locs[N:]
            leading_legs_traj[snapshot_idx] = system.leading_legs
            ts_traj[snapshot_idx] = system.time
            snapshot_idx += 1
            if verbose and (snapshot_idx % 10 == 0):
                print(PROCESS_NAME, snapshot_idx, system.time, T_MAX)
            np.random.seed()

    return np.array(l_sites_traj), np.array(r_sites_traj), np.array(leading_legs_traj), np.array(ts_traj)
