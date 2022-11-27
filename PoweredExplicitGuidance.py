import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class VehicleState:
    radius: float = 0.0                  # radial distance from central body [m]
    altitude: float = 0.0                # distance above ground [m] (radius - R)
    velocity: np.array = np.zeros(3)     # LVLH velocity [m/s], v = [downtrack, crosstrack, radial]
    roll: float = 0.0                    # roll angle [deg]
    pitch: float = 0.0                   # pitch angle [deg]
    yaw: float = 0.0                     # yaw angle [deg]
    isp: float = 0.0                     # specific impluse [s]
    rho: float = 1.15                    # atmospheric density [kg/m^3]
    mass: float = 0.0                    # total vehicle mass [kg]
    mu: float = 3.986004418e14           # gravitational parameter [m^3/s^2] (default is Earth)
    g0: float = 9.81                     # gravitational acceleration [m/s^2] (default is Earth)
    R: float = 6.3781e6                  # radius of central body [m] (default is Earth)
    t: float = 0.0                       # simulation time [s]

    def valid(self):
        if self.radius < 0.0 or self.altitude < self.R or self.mass < 0.0:
            return False

class LaunchSite:
    def CapeCanaveral():
        lat = 28.396837
        lon = -80.605659
        alt = 100
        return lat, lon, alt
    
    def VandenburgAFB():
        lat = 34.75821
        lon = -120.516395
        alt = 100
        return lat, lon, alt

@dataclass
class StageParams:
    mass_flow_rate: float   # [kg/s]
    max_burntime: float     # [s]
    empty_mass: float       # [kg]
    prop_mass: float        # [kg]
    drag_area: float        # [m^2]
    drag_coeff: float       # [-]
    isp: float              # specific impulse [s]

@dataclass
class Vehicle:
    state: VehicleState
    trajectory: list
    first_stage: StageParams
    second_stage: StageParams
    active_stage: StageParams
    launch_site: str = 'Cape Canaveral'

    def __post_init__(self):
        self.active_stage = self.first_stage
        self.set_initial_state()

    def set_initial_state(self):
        if self.launch_site == 'Cape Canaveral':
            lat, lon, alt = LaunchSite.CapeCanaveral()
        elif self.launch_site == 'Vandenburg AFB':
            lat, lon, alt = LaunchSite.VandenburgAFB()
        self.state.altitude = alt
        self.state.radius = self.state.R + alt
        self.state.velocity[0] = (2*np.pi*self.state.R/24/3600)*np.cos(np.deg2rad(lat))
        self.state.pitch = 90
        self.state.mass = self.first_stage.empty_mass + self.first_stage.prop_mass + self.second_stage.empty_mass + self.second_stage.prop_mass
        self.state.isp = self.first_stage.isp
        self.log_state()

    def log_state(self):
        radius = self.state.radius
        altitude = self.state.altitude
        downtrack_vel = self.state.velocity[0]
        radial_vel = self.state.velocity[2]
        pitch = self.state.pitch
        mass = self.state.mass
        time = self.state.t
        state = [radius,altitude,downtrack_vel,radial_vel,pitch,mass,time]
        self.trajectory.append(state)

    def plot_trajectory(self):
        traj = np.array(self.trajectory)
        radius = traj[:,0]
        altitude = traj[:,1]
        downtrack_vel = traj[:,2]
        radial_vel = traj[:,3]
        pitch = traj[:,4]
        mass = traj[:,5]
        time = traj[:,6]

        plt.subplot(231)
        plt.plot(time, radius)
        plt.title('Radius [m]')
        plt.subplot(232)
        plt.plot(time, altitude)
        plt.title('Altitude [m]')
        plt.subplot(233)
        plt.plot(time, downtrack_vel)
        plt.title('Downtrack Velocity [m/s]')
        plt.subplot(234)
        plt.plot(time, radial_vel)
        plt.title('Radial Velocity [m/s]')
        plt.subplot(235)
        plt.plot(time, pitch)
        plt.title('Pitch [deg]')
        plt.subplot(236)
        plt.plot(time, mass)
        plt.title('Mass [kg]')
        plt.suptitle('Ascent Trajectory')
        plt.show()



    def jettison_booster(self):
        self.active_stage = self.second_stage
        self.state.mass = self.state.mass - self.first_stage.empty_mass


@dataclass
class PoweredExplicitGuidanceState:
    T: float = 0.0
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    last_call: float = 0.0
    
@dataclass
class PoweredExplicitGuidance:
    vehicle: Vehicle                         # vehicle data
    peg_state: PoweredExplicitGuidanceState
    min_steering_velocity: float             # minimum velocity magnitude required for PEG [m/s]
    max_pitchrate: float                     # maximum pitchover rate during boost stage [deg/s]
    pitchover_target: float                  # final pitch target during boost stage [deg] 
    passive_pitchover_max_time: float        # maximum time for passive pitchover following booster jettison [s]
    radius_target: float                     # final radius target for PEG [m]
    radial_velocity_target: float            # final radial velocity target for PEG [m/s]
    sim_time_step: float = 0.01             # default 1000Hz sim rate
    guidance_time_step: float = 0.1          # default 10Hz guidance rate

    def ascent_sim(self):
 
        dt = self.sim_time_step
        v_min = self.min_steering_velocity
        pitch_tgt = self.pitchover_target
        max_pitchrate = self.max_pitchrate
        passive_pitchover_max_time = self.passive_pitchover_max_time

        # Boost Stage
        while self.vehicle.state.t < self.vehicle.active_stage.max_burntime:
            #check status of flight  
            v_mag = np.linalg.norm(self.vehicle.state.velocity)  
            pitch = self.vehicle.state.pitch
            if v_mag < v_min :
                #lock steering to UP until velocity > v_init
                self.atmos_active_pitchover_step(0, 90 , 0)
                self.vehicle.log_state()
            elif v_mag >= v_min and pitch > pitch_tgt :
                #pitchover maneuver
                pitchover = pitch - (max_pitchrate * dt)
                self.atmos_active_pitchover_step(0, max(pitchover, pitch_tgt) , 0) 
                self.vehicle.log_state()
            elif v_mag >= v_min and pitch <= pitch_tgt :
                #free gravity turn along prograde vector
                self.atmos_active_step()
                self.vehicle.log_state()
            else:
                break
 
        # jettison booster
        self.vehicle.jettison_booster()
        self.vehicle.log_state()
        t_jettison = self.vehicle.state.t
    
        # initialize PEG
        self.peg_state.T = self.vehicle.active_stage.max_burntime
        self.peg_state.last_call = self.peg_state.T

        [A, B, C, T] = self.poweredExplicitGuidance(initialize=True)
        self.peg_state.A = A
        self.peg_state.B = B
        self.peg_state.C = C
        self.peg_state.T = T
        self.peg_state.last_call = T
 
        # intial PEG pitch angle
        sinpitch = A + C 
        pitch_peg_init = np.rad2deg(np.arcsin(min(1, max(-1, sinpitch)))) 
    
        while self.vehicle.state.pitch != pitch_peg_init:
            if np.abs(self.vehicle.state.pitch - pitch_peg_init) < max_pitchrate*dt:
                break

            if self.vehicle.state.t - t_jettison > passive_pitchover_max_time:
                break
      
            pitch_next = self.vehicle.state.pitch + max_pitchrate*np.sign(pitch_peg_init-self.vehicle.state.pitch)*dt
            self.passive_pitchover_step(0, pitch_next , 0)
            self.vehicle.log_state()
    
        t_final_burn_start = self.vehicle.state.t
        while self.vehicle.state.t < t_final_burn_start + self.vehicle.active_stage.max_burntime :
            if self.peg_state.T <= 0.0: 
                break
            self.peg_step()
            self.vehicle.log_state()


    def atmos_active_step(self):
        dt = self.sim_time_step

        # update mass
        m = self.vehicle.state.mass 
        mdot = self.vehicle.active_stage.mass_flow_rate
        dm = dt * mdot # propellant mass loss
        self.vehicle.state.mass = m - dm

        # update atmosphere dependent states
        isp = self.vehicle.state.isp # TODO: dynamic update based on altitude
        rho = self.vehicle.state.rho # TODO: dynamic update based on altitude
        
        # drag force
        v = self.vehicle.state.velocity
        q = 0.5 * np.dot(v,v) * rho
        area = self.vehicle.active_stage.drag_area
        Cd = self.vehicle.active_stage.drag_coeff # TODO: dynamic update based on velocity
        Fd = -Cd * area * q  

        # thrust force
        g0 = self.vehicle.state.g0
        Ft = isp * g0 * mdot

        # gravity force
        r = self.vehicle.state.radius
        mu = self.vehicle.state.mu
        Fg = - m * mu / (r**2)

        # centrifugal force
        Fc = m * v[0]**2 / r

        # sum of forces
        pitch = self.vehicle.state.pitch
        F_downtrack = (Ft + Fd) * np.cos(np.deg2rad(pitch))
        F_crosstrack = 0.0 # negligable / controllable by RCS
        F_radial = (Ft + Fd) * np.sin(np.deg2rad(pitch)) + Fg + Fc
        F = np.array([F_downtrack, F_crosstrack, F_radial]) # LVLH force vector

        # update velocity
        accel = F / m
        v_next = v + accel * dt # euler integration, TODO: rk4
        self.vehicle.state.velocity = v_next

        # update radius
        r_next = r + v[2] * dt # euler integration, TODO: rk4
        self.vehicle.state.radius = r_next
        self.vehicle.state.altitude = r_next - self.vehicle.state.R

        # update pitch
        pitch_next = np.rad2deg(np.arctan2(v_next[2], v_next[0]))
        self.pitch = pitch_next

        # update time
        t = self.vehicle.state.t
        self.vehicle.state.t = t + dt



    def passive_step(self):
        dt = self.sim_time_step
        m = self.vehicle.state.mass 

        # gravity force
        r = self.vehicle.state.radius
        mu = self.vehicle.state.mu
        Fg = - m * mu / (r**2)

        # centrifugal force
        v = self.vehicle.state.velocity
        Fc = m * v[0]**2 / r

        # sum of forces
        pitch = self.vehicle.state.pitch
        F_downtrack = 0.0
        F_crosstrack = 0.0 
        F_radial = Fg + Fc
        F = np.array([F_downtrack, F_crosstrack, F_radial]) # LVLH force vector

        # update velocity
        accel = F / m
        v_next = v + accel * dt # euler integration, TODO: rk4
        self.vehicle.state.velocity = v_next

        # update radius
        r_next = r + v[2] * dt # euler integration, TODO: rk4
        self.vehicle.state.radius = r_next
        self.vehicle.state.altitude = r_next - self.vehicle.state.R

        # update pitch
        pitch_next = np.rad2deg(np.arctan2(v_next[2], v_next[0]))
        self.pitch = pitch_next

        # update time
        t = self.vehicle.state.t
        self.vehicle.state.t = t + dt



    def atmos_active_pitchover_step(self, roll, pitch, yaw):
        self.vehicle.state.pitch = pitch  
        self.vehicle.state.yaw = yaw
        self.vehicle.state.roll = roll
        self.atmos_active_step()
        self.vehicle.state.pitch = pitch


    def passive_pitchover_step(self, roll, pitch, yaw):
        self.vehicle.state.pitch = pitch  
        self.vehicle.state.yaw = yaw
        self.vehicle.state.roll = roll
        self.passive_step()
        self.vehicle.state.pitch = pitch


    def peg_step(self):
        dt = self.sim_time_step

        # update PEG states at guidance rate
        if (self.peg_state.last_call - self.peg_state.T) >= self.guidance_time_step:
          [A, B, C, T] = self.poweredExplicitGuidance()
          self.peg_state.last_call = T
          self.peg_state.A = A
          self.peg_state.B = B
          self.peg_state.C = C
          self.peg_state.T = T
        else:
          self.peg_state.T = self.peg_state.T - dt

        # update pitch first using PEG equations
        sinpitch = self.peg_state.A - (self.peg_state.last_call - self.peg_state.T) * self.peg_state.B + self.peg_state.C
        pitch_next = np.rad2deg(np.arcsin(min(1, max(-1, sinpitch))))
        self.vehicle.state.pitch = pitch_next 

        # update mass
        m = self.vehicle.state.mass 
        mdot = self.vehicle.active_stage.mass_flow_rate
        dm = dt * mdot # propellant mass loss
        self.vehicle.state.mass = m - dm   

        # thrust force
        isp = self.vehicle.state.isp
        g0 = self.vehicle.state.g0
        Ft = isp * g0 * mdot

        # gravity force
        r = self.vehicle.state.radius
        mu = self.vehicle.state.mu
        Fg = - m * mu / (r**2)

        # centrifugal force
        v = self.vehicle.state.velocity
        Fc = m * v[0]**2 / r

        # sum of forces
        F_downtrack = Ft * np.cos(np.deg2rad(pitch_next))
        F_crosstrack = 0.0 # negligable / controllable by RCS
        F_radial = Ft * np.sin(np.deg2rad(pitch_next)) + Fg + Fc
        F = np.array([F_downtrack, F_crosstrack, F_radial]) # LVLH force vector

        # update velocity
        accel = F / m
        v_next = v + accel * dt # euler integration, TODO: rk4
        self.vehicle.state.velocity = v_next

        # update radius
        r_next = r + v[2] * dt # euler integration, TODO: rk4
        self.vehicle.state.radius = r_next
        self.vehicle.state.altitude = r_next - self.vehicle.state.R

        # update time
        t = self.vehicle.state.t
        self.vehicle.state.t = t + dt


    def poweredExplicitGuidance(self,  initialize=False):
        dt = self.guidance_time_step
        radius = self.vehicle.state.radius
        vt = self.vehicle.state.velocity[0]
        vr = self.vehicle.state.velocity[2]
        tgt = self.radius_target
        acc = self.vehicle.state.isp * self.vehicle.state.g0 * self.vehicle.active_stage.mass_flow_rate / self.vehicle.state.mass
        ve = self.vehicle.state.isp * self.vehicle.state.g0
        oldA = self.peg_state.A
        oldB = self.peg_state.B
        oldT = self.peg_state.T
        mu = self.vehicle.state.mu
        tau = ve / acc

        if initialize:
            # this is to prevent NAN from logarithm because of bad (to high) estimate of T
            if oldT > tau:
                oldT = 0.9*tau
            b0 = -ve*np.log(1-oldT/tau)
            b1 = b0*tau - ve*oldT
            c0 = b0*oldT - b1
            c1 = c0*tau - ve*oldT*oldT/2
            # NOTE for now r_dot target is zero for circular target orbit
            MB = np.array([-vr, tgt - radius - vr*oldT])
            MA = np.array([[b0, b1], [c0, c1]])
            MX = np.linalg.solve(MA, MB)
            A = MX[0]
            B = MX[1]
        else:
            A = oldA
            B = oldB

        # angular momentum
        h_vec = np.cross([0, 0, radius], [vt, 0, vr])
        h = np.linalg.norm(h_vec)
        
        # orbital velocity at a target altitude of tgt [m/s]
        v_tgt = np.sqrt(mu/tgt)

        # zero vertical velocity target
        ht_vec = np.cross([0, 0, tgt], [v_tgt, 0, 0])
        ht = np.linalg.norm(ht_vec)
        dh = ht - h  # angular momentum to gain [unit? m2/s]
        rbar = (radius + tgt)/2  # mean radius [m]

        # Vehicle performence
        C = (mu/(radius*radius) - vt*vt/radius) / acc
        fr = A + C  # sin(pitch) at current time

        # estimation
        CT = (mu/(tgt*tgt) - v_tgt*v_tgt/tgt) / (acc / (1 - oldT/tau))
        frT = A + B*oldT + CT  # sin(pitch) at burnout
        frdot = (frT - fr)/oldT
        ftheta = 1 - fr*fr/2
        fthetadot = -(fr*frdot)
        fthetadotdot = -frdot*frdot/2

        # Ideal velocity-to-gain in order to get target angular momentum, based on current pitch guidance
        dv = (dh/rbar + ve*(oldT-dt)*(fthetadot + fthetadotdot*tau) + fthetadotdot*ve *
              (oldT-dt)*(oldT-dt)/2) / (ftheta + fthetadot*tau + fthetadotdot*tau*tau)

        # Estimate updated burnout time
        T = tau * (1 - np.exp(-dv/ve))

        if T >= 7.5:
            b0 = -ve*np.log(1-T/tau)
            b1 = b0*tau - ve*T
            c0 = b0*T - b1
            c1 = c0*tau - ve*T*T/2
            # for now r_dot target is zero for circular target orbit
            MB = np.array([-vr, tgt - radius - vr*T])
            MA = np.array([[b0, b1], [c0, c1]])
            MX = np.linalg.solve(MA, MB)
            A = MX[0]
            B = MX[1]
        else: # small time remaining to target state
            A = oldA
            B = oldB

        return A, B, C, T


def main():
    stage1 = StageParams(mass_flow_rate=500, max_burntime=150, empty_mass=20000, prop_mass=75000, drag_area=7, drag_coeff=0.2, isp=240)
    stage2 = StageParams(mass_flow_rate=20, max_burntime=300, empty_mass=1500, prop_mass=6000, drag_area=2, drag_coeff=0.2, isp=340)

    inital_vehicle_state = VehicleState()
    vehicle = Vehicle(state=inital_vehicle_state,trajectory=[],first_stage=stage1,second_stage=stage2,active_stage=stage1)

    initial_peg_state = PoweredExplicitGuidanceState()
    peg = PoweredExplicitGuidance(vehicle=vehicle,peg_state=initial_peg_state,min_steering_velocity=800, max_pitchrate=2,pitchover_target=10,passive_pitchover_max_time=30,radius_target=inital_vehicle_state.R+100000,radial_velocity_target=0)
    peg.ascent_sim()
    peg.vehicle.plot_trajectory()
    


if __name__ == '__main__':
    main()
