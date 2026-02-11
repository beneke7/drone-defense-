import numpy as np
from scipy.optimize import linear_sum_assignment
import multiprocessing
import time
import math
import json

# ==========================================
# CONFIGURATION V2
# ==========================================
with open("sim_v2_config.json", "r") as f:
    CONFIG = json.load(f)

FRAMES = CONFIG['simulation']['frames']
DT = CONFIG['simulation']['dt']
MAX_ENTITIES = CONFIG['simulation']['max_entities']
BASE_POS = np.array(CONFIG['simulation'].get('base_pos', [0,0,0]), dtype=np.float32)

# Type IDs
ID_SHAHED = 1
ID_DECOY = 2
ID_KALIBR = 3
ID_KINZHAL = 4
ID_PATRIOT_MSE = 30
ID_GEPARD = 31
ID_IRON_BEAM = 32
ID_EW_JAMMER = 33
ID_INTERCEPTOR_DRONE = 34
ID_OUTPOST = 35
ID_MISSILE_PAC3 = 40
ID_BULLET_35MM = 41

# Events
EVT_NONE = 0
EVT_FIRE = 1
EVT_HIT = 99
EVT_JAMMED = 50
EVT_BASE_HIT = 100 

# Load Config Stats
STATS = {
    ID_PATRIOT_MSE: CONFIG['defenses']['patriot_mse'],
    ID_GEPARD:      CONFIG['defenses']['gepard'],
    ID_IRON_BEAM:   CONFIG['defenses']['iron_beam'],
    ID_EW_JAMMER:   CONFIG['defenses']['ew_jammer'],
    ID_INTERCEPTOR_DRONE: {'range': 20000, 'cost': 2500, 'speed': 80},
    ID_MISSILE_PAC3: CONFIG['projectiles']['pac3_missile'],
    ID_BULLET_35MM:  CONFIG['projectiles']['flak_tracer'],
}

THREAT_STATS = {
    ID_SHAHED:  CONFIG['units']['shahed_136'],
    ID_DECOY:   CONFIG['units']['decoy'],
    ID_KALIBR:  CONFIG['units']['kalibr'],
    ID_KINZHAL: CONFIG['units']['kinzhal'],
}

# ==========================================
# PHYSICS UTILS
# ==========================================
def predict_intercept(shooter_pos, target_pos, target_vel, projectile_speed):
    to_target = target_pos - shooter_pos
    v_t = target_vel
    v_s = projectile_speed
    
    a = np.dot(v_t, v_t) - v_s**2
    b = 2 * np.dot(to_target, v_t)
    c = np.dot(to_target, to_target)
    
    if abs(a) < 1e-6:
        t = -c / (2*b) if b != 0 else 0
    else:
        delta = b**2 - 4*a*c
        if delta < 0: return target_pos # No solution
        t1 = (-b + math.sqrt(delta)) / (2*a)
        t2 = (-b - math.sqrt(delta)) / (2*a)
        if t1 > 0 and t2 > 0: t = min(t1, t2)
        elif t1 > 0: t = t1
        elif t2 > 0: t = t2
        else: return target_pos
        
    return target_pos + (v_t * t)

# ==========================================
# SIMULATION ENTITY
# ==========================================
class SimEntity:
    def __init__(self, uid, type_id, pos, vel=(0,0,0)):
        self.uid = uid
        self.type_id = type_id
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.active = True
        self.target = None # Entity object
        self.target_uid = -1 # ID for Data Log
        self.event_code = EVT_NONE
        
        self.cooldown = 0.0
        self.dwell_timer = 0.0 
        self.jammed = False
        
        # Physics Mode
        self.mode = 'linear'
        if type_id in THREAT_STATS:
            self.mode = THREAT_STATS[type_id].get('physics_mode', 'linear')
        
        self.evasion_phase = np.random.uniform(0, 10)
        
        # Health Init
        if type_id < 10: # Threats
            self.health = 100 
        else:
            self.health = 100

    def update(self, dt, time_total):
        # 1. JAMMED BEHAVIOR
        if self.jammed:
            self.pos += self.vel * dt
            self.vel[1] -= 9.81 * dt # Fall
            return

        # 2. THREAT BEHAVIOR (By Mode)
        if self.type_id < 10: 
            speed = THREAT_STATS[self.type_id]['speed']
            to_base = BASE_POS - self.pos
            dist_base = np.linalg.norm(to_base)
            
            if dist_base > 0:
                base_dir = to_base / dist_base
            else:
                base_dir = np.array([0,0,0])

            if self.mode == 'noe': # Nap of Earth (Decoys)
                # target altitude ~10m
                desired_y = 10
                # Steer towards base on XZ, adjust Y independently
                desired_vel = base_dir * speed
                # Soft ceiling
                current_y = self.pos[1]
                desired_vel[1] = (desired_y - current_y) * 2.0 
                
                self.vel = self.vel * 0.9 + desired_vel * 0.1
                
            elif self.mode == 'jinking': # Kalibr
                # Jink when close (<1000m)
                if dist_base < 3000:
                    jink_factor = math.sin(time_total * 5 + self.evasion_phase) * 0.5
                    # Perpendicular vector? 
                    # Simple jonk: Add Sine to X
                    jink_vec = np.array([math.cos(time_total*3), 0, math.sin(time_total*3)]) * speed * 0.2
                    desired_vel = base_dir * speed + jink_vec
                else:
                    desired_vel = base_dir * speed
                
                self.vel = self.vel * 0.95 + desired_vel * 0.05
                
            elif self.mode == 'ballistic': # Kinzhal
                # Gravity assisted dive
                # It starts with velocity towards base generally
                # Add gravity
                self.vel[1] -= 9.81 * dt
                # Drag? Hypersonic gliders maintain speed.
                
            else: # Linear (Shahed)
                self.vel = base_dir * speed

        # 3. PROJECTILE BEHAVIOR (Guidance)
        elif self.type_id == ID_MISSILE_PAC3 and self.target and self.target.active:
             # ProNav
             # Predict intercept
             intercept = predict_intercept(self.pos, self.target.pos, self.target.vel, STATS[ID_MISSILE_PAC3]['speed'])
             to_int = intercept - self.pos
             dist = np.linalg.norm(to_int)
             if dist > 0:
                desired = (to_int / dist) * STATS[ID_MISSILE_PAC3]['speed']
                # G-Limit?
                self.vel = self.vel * 0.9 + desired * 0.1
        
        # Apply Velocity
        self.pos += self.vel * dt
        self.event_code = EVT_NONE

# ==========================================
# ADVANCED WTA LOGIC
# ==========================================
def calc_cost_matrix_v2(args):
    defenses, drones = args
    n_def = len(defenses)
    n_dro = len(drones)
    cost = np.full((n_def, n_dro), 1e12, dtype=np.float32) 
    
    for i, w in enumerate(defenses):
        p_w = np.array(w['pos'])
        w_type = w['type']
        w_range_sq = w['range'] ** 2
        w_cost = w['cost']
        
        for j, t in enumerate(drones):
            p_t = np.array(t['pos'])
            dist_sq = np.sum((p_w - p_t)**2)
            
            if dist_sq > w_range_sq: continue
            
            # Constraint: Specific Matchups
            if w_type == ID_PATRIOT_MSE and t['type'] < 3: continue 
                
            threat_val = THREAT_STATS.get(t['type'], {'value': 0})['value']
            
            # Estimated Pk
            pk = 0.5
            if w_type == ID_PATRIOT_MSE: pk = STATS[ID_PATRIOT_MSE]['pk']
            elif w_type == ID_GEPARD: pk = STATS[ID_GEPARD]['pk']
            elif w_type == ID_IRON_BEAM: pk = 1.0 
            
            expected_save = threat_val * pk
            score = w_cost - expected_save
            
            cost[i, j] = score + 1e9 
            
    return cost

# ==========================================
# MAIN LOOP
# ==========================================
def run_v2():
    print("Initializing Simulation V2: Mechanics Overhaul")
    
    # 9 Channels
    history = np.zeros((FRAMES, MAX_ENTITIES, 9), dtype=np.float32)
    
    entities = []
    entities_map = {} 
    next_uid = 0
    
    def spawn(type_id, pos, vel=(0,0,0)):
        nonlocal next_uid
        if next_uid >= MAX_ENTITIES: return None
        e = SimEntity(next_uid, type_id, pos, vel)
        entities.append(e)
        entities_map[next_uid] = e
        next_uid += 1
        return e

    # --- SETUP DEFENSES ---
    spawn(ID_PATRIOT_MSE, (0, 0, -2000))
    spawn(ID_IRON_BEAM, (-2000, 0, 0))
    spawn(ID_IRON_BEAM, (2000, 0, 0))
    spawn(ID_GEPARD, (-500, 0, 1000))
    spawn(ID_GEPARD, (500, 0, 1000))
    spawn(ID_GEPARD, (0, 0, 1500))
    spawn(ID_EW_JAMMER, (0, 0, 0))
    
    print("Starting Simulation...")
    
    for f in range(FRAMES):
        sim_time = f * DT
        
        # --- SPAWN TIMELINE ---
        seed_target = np.array([0,0,0]) # Base
        
        if f == 1: # Saturation
             for i in range(20):
                 pos = np.array([np.random.uniform(-3000,3000), 200, np.random.uniform(5000, 6000)])
                 spawn(ID_SHAHED, pos, (0,0,-10)) # Initial push vector
                 
             for i in range(50):
                 pos = np.array([np.random.uniform(-4000,4000), 300, np.random.uniform(5000, 7000)])
                 spawn(ID_DECOY, pos, (0,0,-10))

        if f == 1500: # Kalibrs
            for i in range(5):
                pos = np.array([np.random.uniform(-4000,4000), 100, 9000])
                spawn(ID_KALIBR, pos, (0,0,-100))
                      
        if f == 3000: # Kinzhals
            for i in range(2):
                pos = np.array([np.random.uniform(-500,500), 10000, 5000]) # High Altitude
                # Dive Vector
                vel = (seed_target - pos)
                vel = vel / np.linalg.norm(vel) * THREAT_STATS[ID_KINZHAL]['speed']
                spawn(ID_KINZHAL, pos, vel)

        # --- UPDATE & LOGIC ---
        active_threats = [e for e in entities if e.active and e.type_id < 10]
        # Defenses excludes projectiles
        active_defenses = [e for e in entities if e.active and 30 <= e.type_id < 40] 

        # 1. JAMMER LOGIC
        jammers = [d for d in active_defenses if d.type_id == ID_EW_JAMMER]
        for jam in jammers:
            jam_range_sq = STATS[ID_EW_JAMMER]['range']**2
            for t in active_threats:
                if t.jammed: continue
                dist_sq = np.sum((jam.pos - t.pos)**2)
                if dist_sq < jam_range_sq:
                    guidance = THREAT_STATS.get(t.type_id, {}).get('guidance', 'simple')
                    prob = STATS[ID_EW_JAMMER]['prob_gps'] if guidance == 'gps' else STATS[ID_EW_JAMMER]['prob_ins']
                    if np.random.random() < prob * DT:
                        t.jammed = True
                        t.vel += np.random.uniform(-10, 10, 3) 
                        t.event_code = EVT_JAMMED

        # 2. WTA (10Hz)
        shooting_defenses = [d for d in active_defenses if d.type_id != ID_EW_JAMMER]
        
        if f % 10 == 0 and active_threats and shooting_defenses:
            def_dat = [{'pos': d.pos, 'type': d.type_id, 'range': STATS[d.type_id]['range'], 'cost': STATS[d.type_id]['cost']} for d in shooting_defenses]
            thr_dat = [{'pos': t.pos, 'type': t.type_id} for t in active_threats]
            
            cost_mat = calc_cost_matrix_v2((def_dat, thr_dat))
            
            try:
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                for r, c in zip(row_ind, col_ind):
                    if cost_mat[r, c] < 1e11:
                        defense = shooting_defenses[r]
                        target = active_threats[c]
                        
                        if defense.cooldown <= 0:
                            # ACTION: FIRE
                            defense.cooldown = STATS[defense.type_id]['reload_time']
                            defense.event_code = EVT_FIRE
                            defense.target_uid = target.uid
                            
                            if defense.type_id == ID_PATRIOT_MSE:
                                speed = STATS[ID_MISSILE_PAC3]['speed']
                                proj = spawn(ID_MISSILE_PAC3, defense.pos + [0,10,0], [0, speed*0.5, 0])
                                if proj: proj.target = target
                                
                            elif defense.type_id == ID_GEPARD:
                                # Spawn Hitscan Tracer WITH PREDICTION
                                speed = STATS[ID_BULLET_35MM]['speed']
                                intercept = predict_intercept(defense.pos, target.pos, target.vel, speed)
                                
                                to_int = intercept - defense.pos
                                dist = np.linalg.norm(to_int)
                                if dist > 0:
                                    vel = (to_int / dist) * speed
                                    for _ in range(3): # Burst
                                        spawn(ID_BULLET_35MM, defense.pos + [0,2,0], vel + np.random.uniform(-10,10,3))
                                        
                                # Roll Hit Logic
                                if np.random.random() < STATS[ID_GEPARD]['pk']:
                                    target.health -= 35
                                    if target.health <= 0:
                                        target.active = False
                                        target.event_code = EVT_HIT

                            elif defense.type_id == ID_IRON_BEAM:
                                defense.target = target
                                defense.dwell_timer = 0.0
            except ValueError: pass

        # 3. ENTITY UPDATES
        for e in entities:
             if not e.active: continue
             
             e.update(DT, sim_time)
             e.cooldown -= DT
             
             # Base Hit
             if e.type_id < 10 and not e.jammed:
                 dist_base = np.linalg.norm(e.pos - BASE_POS)
                 if dist_base < 50:
                     e.active = False
                     e.event_code = EVT_BASE_HIT
             
             # PAC-3 Hit Check
             if e.type_id == ID_MISSILE_PAC3 and e.target:
                 if not e.target.active: 
                     e.active = False
                 else:
                     dist = np.linalg.norm(e.target.pos - e.pos)
                     if dist < 50: # HIT
                         e.active = False
                         e.target.active = False
                         e.target.event_code = EVT_HIT

             # Iron Beam Logic
             if e.type_id == ID_IRON_BEAM and e.target:
                 if not e.target.active or np.linalg.norm(e.target.pos - e.pos) > STATS[ID_IRON_BEAM]['range']:
                     e.target = None
                     e.target_uid = -1
                 else:
                     e.dwell_timer += DT
                     e.target_uid = e.target.uid
                     if e.dwell_timer >= STATS[ID_IRON_BEAM]['dwell_time']:
                         e.target.active = False
                         e.target.event_code = EVT_HIT
                         e.target = None
                         e.target_uid = -1
                         e.dwell_timer = 0
             else:
                 e.target_uid = -1 

             # De-spawn floor or far
             if e.pos[1] < -10 or np.linalg.norm(e.pos) > 50000: e.active = False

             # Data Logging
             if e.uid < MAX_ENTITIES:
                 history[f, e.uid, 0] = 1.0
                 history[f, e.uid, 1] = e.type_id
                 history[f, e.uid, 2:5] = e.pos
                 history[f, e.uid, 7] = e.event_code
                 history[f, e.uid, 8] = e.target_uid

    print(f"Loop Complete. Saving...")
    np.save("simulation_data_v2.npy", history)

if __name__ == '__main__':
    run_v2()
