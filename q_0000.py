# -*- coding: utf-8 -*-
import qutip as qt
import numpy as np
import math
import json
import os
import sys
import importlib
from collections import deque

# ============================================================
# КОНСТАНТЫ (планковские единицы: ħ = c = G = 1)
# ============================================================

L_PL = 1.0
R_INNER = L_PL
R_OUTER = 2 * L_PL
R_SENSOR = 3 * L_PL
GROWTH_THRESHOLD = 10 * L_PL
DISSIPATION_RATE = L_PL
MERGE_DISTANCE = R_OUTER
DIPOLE_LEN_INIT = 6 * L_PL
DIPOLE_LEN_MIN = L_PL
COMPRESSION_STEP = L_PL
CHAIN_MERGE_DISTANCE = R_OUTER
CHAIN_BINDING_FACTOR = 0.5

Z_MAX = 6 * L_PL
Z_MIN = 1 * L_PL
E_MAX_QUANTUM = 1.0
E_THRESHOLD_EMIT = 0.8

C_LIGHT_VAL = 1.0
G_CONST = 1.0
HBAR_VAL = 1.0
K_B = 1.0
EPS0_VAL = 1.0
MU0_VAL = 1.0

L_PLANCK = 1.0
M_PLANCK = 1.0
T_PLANCK = 1.0
TEMP_PLANCK = 1.0
Q_PLANCK = 1.0

E_PLANCK = 1.0
V_PLANCK = 1.0
I_PLANCK = 1.0
B_PLANCK = 1.0
PHI_PLANCK = 1.0
EMF_PLANCK = 1.0

GRAVITY_SIGN = -1.0

CACHE_BEGIN_MARKER = "# === CACHE BEGIN ==="
CACHE_END_MARKER = "# === CACHE END ==="


# ============================================================
# ОТСТРЕЛЯННЫЙ ЭЛЕКТРОН
# ============================================================

class DischargedCharge:
    # Отстрелянный электрон — квант разности потенциалов.
    # Детерминирован: одна энергия, один знак, один акт передачи.
    def __init__(self, parent_id, pos, sign=-1.0):
        self.parent_id = parent_id
        self.pos = pos
        self.sign = sign
        self.length = L_PL
        self.N = 1.0
        self.energy = E_PLANCK / self.N
        self.voltage = V_PLANCK / self.N
        self.alive = True
        self.hits = 0
        self.velocity = (0.0, 0.0, 0.0)

    def collide(self, target_energy_absorbed=0.0):
        if not self.alive:
            return None
        self.hits += 1
        given = self.energy
        self.alive = False
        return {
            'parent': self.parent_id,
            'energy_given': given,
            'sign': self.sign,
            'hits': self.hits,
        }

    def report(self):
        return {
            'parent_id': self.parent_id,
            'pos': self.pos,
            'sign': self.sign,
            'length_LP': self.length / L_PL,
            'N': self.N,
            'energy_J': self.energy,
            'voltage_V': self.voltage,
            'alive': self.alive,
            'hits': self.hits,
        }


# ============================================================
# ПЛАНКОВСКИЙ КВАНТ
# ============================================================

class PlanckQuantum:
    """
    Планковский квант — три независимых центра (X, Y, Z).
    Раздельные move_* для каждого центра.
    Намёк на слияние: аннигиляция X и Y оставляет Z.
    """

    def __init__(self, idx, spin_dir=1, charge_sign=1, generation=0):
        self.id = idx
        self.spin = float(spin_dir)
        self.charge = float(charge_sign)
        self.generation = float(generation)

        self.E_x = 0.0
        self.c_y = 0.0
        self.m_z = 0.0
        self.energy_total = 0.0
        self.delta = 0.0
        self.temperature = 0.0
        self.growth = 0.0
        self.cycle_count = 0

        self.model_flip_count = 0
        self.flip_cycles = []
        self.model_orientation = 1.0

        self.spin_flip_count = 0
        self.spin_phase = 0.0

        self.is_zero_state = True

        self.center_X = (0.0, 0.0, 0.0)
        self.center_Y = (0.0, 0.0, 0.0)
        self.center_Z = (0.0, 0.0, 0.0)

        self.pos_X = 0.0
        self.pos_Y = 0.0
        self.pos_Z = 0.0

        self.orient_X = 1.0
        self.orient_Y = 1.0
        self.orient_Z = 1.0

        self.X_dipole_len = 6 * L_PL
        self.Y_dipole_len = 6 * L_PL
        self.Z_dipole_len = DIPOLE_LEN_INIT
        self.Z_compression = 0.0

        self.X_positive_arm = 3 * L_PL
        self.X_negative_arm = 3 * L_PL
        self.X_asymmetry = 0.0
        self.electron_count = 0
        self.emitted_charges = []
        self.luminosity = 0.0

        self.orbital_count = 0.0
        self.dynamic_energy = 0.0
        self.potential_energy = 0.0

        self.sensor = {
            'X+': L_PL, 'X-': L_PL,
            'Y+': L_PL, 'Y-': L_PL,
            'Z+': L_PL, 'Z-': L_PL
        }
        self.stored_sources_X = None
        self.stored_sources_Y = None
        self.stored_sources_Z = None

        self.alive = True
        self.detach_count = 0
        self.children = []
        self.merged = False
        self.partner = None
        self.merge_energy = 0.0
        self.merge_potential_diff = 0.0
        self.merge_stage = 0
        self.merge_orientation_Y = 0.0
        self.merge_orientation_Z = 0.0

        self.chain_linked = False
        self.chain_next = None
        self.chain_prev = None
        self.chain_energy = 0.0
        self.chain_potential_diff = 0.0
        self.chain_length = 1
        self.quanta_ref = []

        self.planck_energy_X = 0.0
        self.planck_energy_Y = 0.0
        self.planck_energy_Z = 0.0
        self.planck_voltage_X = 0.0
        self.planck_voltage_Y = 0.0
        self.planck_voltage_Z = 0.0
        self.planck_magnetic_Y = 0.0
        self.planck_emf_Y = 0.0
        self.planck_energy_total = 0.0
        self.planck_mass = 0.0
        self.planck_temp = 0.0

        self.magnetic_closed = False
        self.planck_magnetic_internal = 0.0
        self.planck_magnetic_external = 0.0

        self.is_cross = False
        self.is_vortex = True
        self.flicker_phase = 0.0
        self.standing_wave = False
        self.wave_dim = None

        self.last_field_sign = 0.0
        self.last_field_amplitude = 0.0

        self.gravity_sign = GRAVITY_SIGN
        self.planck_gravitational_potential = 0.0

        self.E_times_N_X = 0.0
        self.E_times_N_Y = 0.0
        self.E_times_N_Z = 0.0
        self.E_times_N_total = 0.0
        self.n_points = 0

        self.Z_plus_center = True
        self.Z_minus_orbital = False
        self.Z_orbital_radius = 6 * L_PL
        self.gravitational_wells = 0
        self.wells_per_revolution = 1

        self.spin_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        self.density_matrix = qt.ket2dm(self.spin_state)

        self.reproduction_potential_diff = 0.0

    def diff(self, a, b): return abs(a - b)
    def diff_signed(self, a, b): return a - b
    def sum_diff(self, a, b): return a + b

    def compare_diff(self, value, threshold):
        if value > threshold: return 1.0
        if value < threshold: return -1.0
        return 0.0

    def sign_diff(self, value): return self.compare_diff(value, 0.0)
    def abs_diff(self, value): return abs(value)

    def diff3D(self, point_a, point_b):
        return np.sqrt(
            (point_a[0] - point_b[0]) ** 2 +
            (point_a[1] - point_b[1]) ** 2 +
            (point_a[2] - point_b[2]) ** 2)

    def sync_pos_from_centers(self):
        self.pos_X = (self.center_X[0] + self.center_Y[0] + self.center_Z[0]) / 3.0
        self.pos_Y = (self.center_X[1] + self.center_Y[1] + self.center_Z[1]) / 3.0
        self.pos_Z = (self.center_X[2] + self.center_Y[2] + self.center_Z[2]) / 3.0

    def point_X(self, sign):
        orient = self.orient_X
        abs_sign = self.abs_diff(sign)
        dist = R_SENSOR
        if abs_sign == 1.0: dist = R_INNER
        elif abs_sign == 2.0: dist = R_OUTER
        s_orient = self.sign_diff(orient)
        s_sign = self.sign_diff(sign)
        cx = self.center_X[0]
        if s_orient == s_sign: return cx + dist
        if s_orient == -s_sign: return cx - dist
        return cx

    def point_Y(self, sign):
        orient = self.orient_Y
        abs_sign = self.abs_diff(sign)
        dist = R_SENSOR
        if abs_sign == 1.0: dist = R_INNER
        elif abs_sign == 2.0: dist = R_OUTER
        s_orient = self.sign_diff(orient)
        s_sign = self.sign_diff(sign)
        cy = self.center_Y[1]
        if s_orient == s_sign: return cy + dist
        if s_orient == -s_sign: return cy - dist
        return cy

    def point_Z(self, sign):
        orient = self.orient_Z
        abs_sign = self.abs_diff(sign)
        if abs_sign == 1.0: offset = self.Z_dipole_len / 6.0
        elif abs_sign == 2.0: offset = self.Z_dipole_len / 3.0
        else: offset = self.Z_dipole_len / 2.0
        s_orient = self.sign_diff(orient)
        s_sign = self.sign_diff(sign)
        cz = self.center_Z[2]
        if s_orient == s_sign: return cz + offset
        if s_orient == -s_sign: return cz - offset
        return cz

    def inner_X_pos(self): return (self.point_X(1), self.center_X[1], self.center_X[2])
    def outer_X_pos(self): return (self.point_X(2), self.center_X[1], self.center_X[2])
    def sensor_X_pos(self): return (self.point_X(3), self.center_X[1], self.center_X[2])
    def inner_X_neg(self): return (self.point_X(-1), self.center_X[1], self.center_X[2])
    def outer_X_neg(self): return (self.point_X(-2), self.center_X[1], self.center_X[2])
    def sensor_X_neg(self): return (self.point_X(-3), self.center_X[1], self.center_X[2])

    def inner_Y_pos(self): return (self.center_Y[0], self.point_Y(1), self.center_Y[2])
    def outer_Y_pos(self): return (self.center_Y[0], self.point_Y(2), self.center_Y[2])
    def sensor_Y_pos(self): return (self.center_Y[0], self.point_Y(3), self.center_Y[2])
    def inner_Y_neg(self): return (self.center_Y[0], self.point_Y(-1), self.center_Y[2])
    def outer_Y_neg(self): return (self.center_Y[0], self.point_Y(-2), self.center_Y[2])
    def sensor_Y_neg(self): return (self.center_Y[0], self.point_Y(-3), self.center_Y[2])

    def inner_Z_pos(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(1))
    def outer_Z_pos(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(2))
    def sensor_Z_pos(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(3))
    def inner_Z_neg(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(-1))
    def outer_Z_neg(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(-2))
    def sensor_Z_neg(self): return (self.center_Z[0], self.center_Z[1], self.point_Z(-3))

    def emit_X_sources(self):
        return [
            (self.inner_X_pos(), self.charge), (self.outer_X_pos(), self.charge),
            (self.sensor_X_pos(), self.charge), (self.inner_X_neg(), self.charge),
            (self.outer_X_neg(), self.charge), (self.sensor_X_neg(), self.charge)]

    def emit_Y_sources(self):
        return [
            (self.inner_Y_pos(), self.charge), (self.outer_Y_pos(), self.charge),
            (self.sensor_Y_pos(), self.charge), (self.inner_Y_neg(), self.charge),
            (self.outer_Y_neg(), self.charge), (self.sensor_Y_neg(), self.charge)]

    def emit_Z_sources(self):
        return [
            (self.inner_Z_pos(), self.charge), (self.outer_Z_pos(), self.charge),
            (self.sensor_Z_pos(), self.charge), (self.inner_Z_neg(), self.charge),
            (self.outer_Z_neg(), self.charge), (self.sensor_Z_neg(), self.charge)]

    def get_potential_diff_X(self):
        return self.diff_signed(self.sensor['X+'], self.sensor['X-'])
    def get_potential_diff_Y(self):
        return self.diff_signed(self.sensor['Y+'], self.sensor['Y-'])
    def get_potential_diff_Z(self):
        return self.diff_signed(self.sensor['Z+'], self.sensor['Z-'])
    def get_abs_potential_diff_X(self): return abs(self.get_potential_diff_X())
    def get_abs_potential_diff_Y(self): return abs(self.get_potential_diff_Y())
    def get_abs_potential_diff_Z(self): return abs(self.get_potential_diff_Z())
    def get_total_potential_diff(self):
        return (self.get_abs_potential_diff_X() +
                self.get_abs_potential_diff_Y() +
                self.get_abs_potential_diff_Z())

    def get_potential_diff_XY(self):
        return self.diff3D(self.center_X, self.center_Y)
    def get_potential_diff_YZ(self):
        return self.diff3D(self.center_Y, self.center_Z)
    def get_potential_diff_ZX(self):
        return self.diff3D(self.center_Z, self.center_X)
    def get_total_centers_potential_diff(self):
        return (self.get_potential_diff_XY() +
                self.get_potential_diff_YZ() +
                self.get_potential_diff_ZX())

    def get_potential_energy_compression(self):
        return self.diff(DIPOLE_LEN_INIT, self.Z_dipole_len)

    def get_potential_diff_ECM(self):
        diff_xy = self.diff(self.E_x, self.c_y)
        diff_yz = self.diff(self.c_y, self.m_z)
        diff_zx = self.diff(self.m_z, self.E_x)
        return diff_xy + diff_yz + diff_zx

    def get_potential_diff_field(self, sensor_point, external_sources):
        if not external_sources:
            return L_PL
        potentials = []
        for src_point, src_charge in external_sources:
            distance = self.diff3D(sensor_point, src_point)
            if distance > 0:
                potential = src_charge / distance
            else:
                potential = L_PL
            if self.charge == src_charge:
                potentials.append(potential)
            else:
                potentials.append(-potential)
        if potentials:
            return np.mean(potentials)
        return L_PL

    def get_potential_diff_chain_X(self, other):
        return self.diff(self.sensor['X+'], other.sensor['X-']) + \
               self.diff(self.sensor['X-'], other.sensor['X+'])
    def get_potential_diff_chain_Y(self, other):
        return self.diff(self.sensor['Y+'], other.sensor['Y-']) + \
               self.diff(self.sensor['Y-'], other.sensor['Y+'])
    def get_potential_diff_chain_Z(self, other):
        return self.diff(self.sensor['Z+'], other.sensor['Z-']) + \
               self.diff(self.sensor['Z-'], other.sensor['Z+'])
    def get_total_potential_diff_chain(self, other):
        return (self.get_potential_diff_chain_X(other) +
                self.get_potential_diff_chain_Y(other) +
                self.get_potential_diff_chain_Z(other))
    def get_potential_diff_chain_connect_X(self, other):
        return self.diff(self.sensor['X-'], other.sensor['X+'])
    def get_potential_diff_chain_connect_Y(self, other):
        return self.diff(self.sensor['Y-'], other.sensor['Y+'])
    def get_potential_diff_chain_connect_Z(self, other):
        return self.diff(self.sensor['Z-'], other.sensor['Z+'])
    def get_total_potential_diff_chain_connect(self, other):
        return (self.get_potential_diff_chain_connect_X(other) +
                self.get_potential_diff_chain_connect_Y(other) +
                self.get_potential_diff_chain_connect_Z(other))

    def get_potential_diff_same_charge_X(self, other):
        return self.diff(self.sensor['X+'], other.sensor['X+']) + \
               self.diff(self.sensor['X-'], other.sensor['X-'])
    def get_potential_diff_same_charge_Y(self, other):
        return self.diff(self.sensor['Y+'], other.sensor['Y+']) + \
               self.diff(self.sensor['Y-'], other.sensor['Y-'])
    def get_potential_diff_same_charge_Z(self, other):
        return self.diff(self.sensor['Z+'], other.sensor['Z+']) + \
               self.diff(self.sensor['Z-'], other.sensor['Z-'])
    def get_total_potential_diff_same_charge(self, other):
        if other is None: return 0.0
        return (self.get_potential_diff_same_charge_X(other) +
                self.get_potential_diff_same_charge_Y(other) +
                self.get_potential_diff_same_charge_Z(other))

    def get_chain_binding_energy(self, other):
        E_self = self.energy_total
        E_other = other.energy_total
        same_diff = self.get_total_potential_diff_same_charge(other)
        E_bind_chain = E_self + E_other - same_diff
        E_bind_chain = E_bind_chain * CHAIN_BINDING_FACTOR
        return E_bind_chain

    def get_chain_potential_diff_binding(self, other):
        same_diff = self.get_total_potential_diff_same_charge(other)
        chain_diff = self.get_total_potential_diff_chain(other)
        return self.diff(same_diff, chain_diff)

    def get_potential_diff_merge_X(self, other):
        return self.diff_signed(self.E_x, other.E_x)
    def get_potential_diff_merge_Y(self, other):
        return self.diff_signed(self.c_y, other.c_y)
    def get_potential_diff_merge_Z(self, other):
        return self.diff_signed(self.m_z, other.m_z)
    def get_abs_potential_diff_merge_X(self, other):
        return abs(self.get_potential_diff_merge_X(other))
    def get_abs_potential_diff_merge_Y(self, other):
        return abs(self.get_potential_diff_merge_Y(other))
    def get_abs_potential_diff_merge_Z(self, other):
        return abs(self.get_potential_diff_merge_Z(other))
    def get_total_potential_diff_merge(self, other):
        return (self.get_abs_potential_diff_merge_X(other) +
                self.get_abs_potential_diff_merge_Y(other) +
                self.get_abs_potential_diff_merge_Z(other))
    def get_potential_diff_merge_charge(self, other):
        return self.diff(self.charge, other.charge)
    def get_potential_diff_merge_spin(self, other):
        return self.diff(self.spin, other.spin)
    def get_potential_diff_dark_matter(self, other):
        return (self.get_potential_diff_merge_charge(other) +
                self.get_potential_diff_merge_spin(other))

    def get_potential_diff_magnetic_lock_Y(self, other):
        return self.diff(self.sensor['Y+'], other.sensor['Y-']) + \
               self.diff(self.sensor['Y-'], other.sensor['Y+'])
    def get_potential_diff_magnetic_lock_Z(self, other):
        return self.diff(self.sensor['Z+'], other.sensor['Z-']) + \
               self.diff(self.sensor['Z-'], other.sensor['Z+'])
    def get_total_magnetic_lock_potential_diff(self, other):
        return (self.get_potential_diff_magnetic_lock_Y(other) +
                self.get_potential_diff_magnetic_lock_Z(other))

    def get_potential_diff_reproduction(self, parent_charge, parent_spin):
        q_diff = self.diff_signed(parent_charge, -parent_charge)
        V_charge = self.abs_diff(q_diff)
        s_diff = self.diff_signed(parent_spin, -parent_spin)
        V_spin = self.abs_diff(s_diff)
        return V_charge + V_spin

    def get_charge_potential_diff(self, charge):
        return self.charge * self.abs_diff(charge)
    def get_spin_potential_diff(self, spin):
        return self.spin * self.abs_diff(spin)

    def feel_direction(self, sensor_point, external_sources):
        return self.get_potential_diff_field(sensor_point, external_sources)

    def feel_all_directions(self, external_X=None, external_Y=None, external_Z=None):
        self.sensor['X+'] = self.feel_direction(self.sensor_X_pos(), external_X)
        self.sensor['X-'] = self.feel_direction(self.sensor_X_neg(), external_X)
        self.sensor['Y+'] = self.feel_direction(self.sensor_Y_pos(), external_Y)
        self.sensor['Y-'] = self.feel_direction(self.sensor_Y_neg(), external_Y)
        self.sensor['Z+'] = self.feel_direction(self.sensor_Z_pos(), external_Z)
        self.sensor['Z-'] = self.feel_direction(self.sensor_Z_neg(), external_Z)
        self.update_ECM()

    def update_sensors(self):
        self.feel_all_directions(
            self.stored_sources_X,
            self.stored_sources_Y,
            self.stored_sources_Z)

    def update_ECM(self):
        self.Z_compression = self.get_potential_energy_compression()
        self.E_x = self.get_abs_potential_diff_X()
        self.c_y = self.get_abs_potential_diff_Y()
        sensor_diff_Z = self.get_abs_potential_diff_Z()
        # m_z = только реальная разность Z-сенсоров от поля.
        # Сжатие Z-диполя НЕ добавляется в m_z,
        # иначе положительная обратная связь и нарушение сохранения энергии.
        self.m_z = sensor_diff_Z
        self.dynamic_energy = self.get_abs_potential_diff_X() + self.get_abs_potential_diff_Y()
        self.potential_energy = self.m_z
        self.energy_total = self.E_x + self.c_y + self.m_z
        self.temperature = self.diff(self.energy_total, self.m_z)
        self.delta = self.diff(self.E_x, self.m_z + self.c_y)
        self.calibrate_to_planck()
        self.update_Z_from_energy()
        self.is_zero_state = (self.energy_total <= 0.0)

    def update_Z_from_energy(self):
        if E_MAX_QUANTUM > 0.0:
            e_frac = self.energy_total / E_MAX_QUANTUM
        else:
            e_frac = 0.0
        if e_frac < 0.0: e_frac = 0.0
        if e_frac > 1.0: e_frac = 1.0
        self.Z_dipole_len = Z_MAX - (Z_MAX - Z_MIN) * e_frac
        self.Z_compression = Z_MAX - self.Z_dipole_len

    def energy_fraction(self):
        if E_MAX_QUANTUM > 0.0:
            return self.energy_total / E_MAX_QUANTUM
        return 0.0

    def pull_X(self): return self.get_potential_diff_X()
    def pull_Y(self): return self.get_potential_diff_Y()
    def pull_Z(self): return self.get_potential_diff_Z()

    def has_external_field_int(self):
        for key in self.sensor:
            if self.diff(self.sensor[key], L_PL) != 0.0: return 1.0
        return 0.0

    def has_external_field(self): return self.has_external_field_int() != 0.0

    def spin_frequency(self):
        if L_PL > 0.0: return self.c_y / L_PL
        return 0.0

    def z_pulse_frequency(self):
        return 2.0 * self.spin_frequency()

    def flip_frequency(self):
        if self.c_y > 0.0: return self.c_y / L_PL
        return 0.0

    def orbital_frequency(self):
        v = C_LIGHT_VAL
        r = self.Z_orbital_radius
        if r > 0: return v / (2 * math.pi * r)
        return 0.0

    def gravitational_well_frequency(self):
        return self.orbital_frequency()

    def _null_move(self, *args, **kwargs):
        pass

    def move_center_X(self):
        pull_x = self.pull_X()
        dir_x = self.sign_diff(pull_x)
        cx, cy, cz = self.center_X
        if dir_x == 1.0: cx += L_PL
        elif dir_x == -1.0: cx -= L_PL
        self.center_X = (cx, cy, cz)

    def move_center_Y(self):
        pull_y = self.pull_Y()
        dir_y = self.sign_diff(pull_y)
        cx, cy, cz = self.center_Y
        if dir_y == 1.0: cy += L_PL
        elif dir_y == -1.0: cy -= L_PL
        self.center_Y = (cx, cy, cz)

    def move_center_Z(self):
        pull_z = self.pull_Z()
        dir_z = self.sign_diff(pull_z)
        cx, cy, cz = self.center_Z
        if dir_z == 1.0: cz += L_PL
        elif dir_z == -1.0: cz -= L_PL
        self.center_Z = (cx, cy, cz)

    def update_spin_phase(self):
        diff_XY = self.diff(self.E_x, self.c_y)
        self.spin_phase += diff_XY
        while self.spin_phase >= L_PL:
            self.spin = -self.spin
            if self.spin > 0.0:
                self.spin_state = qt.basis(2, 0)
            else:
                self.spin_state = qt.basis(2, 1)
            self.density_matrix = qt.ket2dm(self.spin_state)
            self.spin_flip_count += 1
            self.spin_phase -= L_PL

    def move(self):
        if self.merged: return None
        pull_x = self.pull_X()
        pull_y = self.pull_Y()
        pull_z = self.pull_Z()
        has_pull = self.sign_diff(self.get_total_potential_diff())
        if has_pull == 1.0: self.orbital_count += L_PL
        self.update_ECM()
        self.update_X_asymmetry()
        self.compress_Z()
        self.update_spin_phase()
        child = self.model_flip()
        for other in self.quanta_ref:
            if other is not self and other.alive and not other.merged:
                if self.can_chain_merge(other):
                    self.try_chain_merge(other)
                    break
        for other in self.quanta_ref:
            if other is not self and other.alive and not other.merged:
                if self.can_merge(other):
                    self.try_merge(other)
                    break
        if not self.merged:
            self.move_center_X()
            self.move_center_Y()
            self.move_center_Z()
            self.sync_pos_from_centers()
            self.growth += self.delta
            if not self.has_external_field():
                self.dissipate()
        self.cycle_count += 1
        if self.Z_dipole_len <= L_PL:
            return self.try_reproduce()
        return child

    def dissipate(self):
        if self.E_x > 0: self.E_x = max(0, self.E_x - DISSIPATION_RATE)
        if self.c_y > 0: self.c_y = max(0, self.c_y - DISSIPATION_RATE)
        if self.m_z > 0: self.m_z = max(0, self.m_z - DISSIPATION_RATE)
        if self.orbital_count > 0: self.orbital_count = max(0, self.orbital_count - L_PL)
        if self.E_x == 0 and self.c_y == 0 and self.m_z == 0:
            for key in self.sensor:
                self.sensor[key] = L_PL
            self.energy_total = 0.0
            self.delta = 0.0
            self.temperature = 0.0
            self.Z_dipole_len = DIPOLE_LEN_INIT
            self.Z_compression = 0.0
            self.growth = 0.0
            self.dynamic_energy = 0.0
            self.potential_energy = 0.0
            self.orbital_count = 0.0
            self.Y_dipole_len = 6 * L_PL
            self.X_positive_arm = 3 * L_PL
            self.X_negative_arm = 3 * L_PL
            self.X_asymmetry = 0.0
            self.spin_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            self.density_matrix = qt.ket2dm(self.spin_state)
            self.is_zero_state = True
        else:
            self.sensor['X+'] = L_PL + self.E_x
            self.sensor['X-'] = L_PL
            self.sensor['Y+'] = L_PL + self.c_y
            self.sensor['Y-'] = L_PL
            self.sensor['Z+'] = L_PL + self.m_z
            self.sensor['Z-'] = L_PL
            dm_array = self.density_matrix.full().copy()
            off_diagonal = dm_array[0, 1]
            dm_array[0, 1] = off_diagonal * np.exp(-L_PL)
            dm_array[1, 0] = off_diagonal * np.exp(-L_PL)
            self.density_matrix = qt.Qobj(dm_array, dims=self.density_matrix.dims)
            self.update_ECM()
        self.growth = self.diff(self.growth, DISSIPATION_RATE)

    def update_X_asymmetry(self):
        if 6 * L_PL > 0.0:
            Z_ratio = (6 * L_PL - self.Z_dipole_len) / (6 * L_PL)
        else:
            Z_ratio = 0.0
        self.X_asymmetry = Z_ratio
        self.X_positive_arm = 3 * L_PL * (1.0 - Z_ratio * 0.5)
        self.X_negative_arm = 3 * L_PL * (1.0 + Z_ratio)
        self.X_dipole_len = self.X_positive_arm + self.X_negative_arm
        self.luminosity = self.spin_frequency()
        ASYMMETRY_THRESHOLD = 0.8
        if Z_ratio > ASYMMETRY_THRESHOLD:
            return self.emit_charge()
        return None

    def emit_charge(self):
        sign = -1.0 if self.charge > 0 else 1.0
        discharged = DischargedCharge(
            parent_id=self.id,
            pos=(self.center_X[0] - self.X_negative_arm,
                 self.center_X[1], self.center_X[2]),
            sign=sign)
        self.emitted_charges.append(discharged)
        self.electron_count += 1
        self.X_negative_arm = 3 * L_PL
        diff_XY = self.diff(self.E_x, self.c_y)
        self.spin_phase += diff_XY
        return discharged

    def compress_Z(self):
        if self.energy_total < GROWTH_THRESHOLD: return False
        if self.Z_dipole_len <= DIPOLE_LEN_MIN: return False
        self.Z_dipole_len = self.diff(self.Z_dipole_len, COMPRESSION_STEP)
        self.growth = 0.0
        return True

    def model_flip(self):
        pull_x = self.pull_X()
        pull_y = self.pull_Y()
        sx = self.sign_diff(pull_x)
        sy = self.sign_diff(pull_y)

        if self.Z_dipole_len <= DIPOLE_LEN_MIN and sx != 0.0 and sy != 0.0 and sx == sy:
            self.orient_Y = -self.orient_Y
            self.orient_Z = -self.orient_Z
            self.spin = -self.spin
            if self.spin > 0.0:
                self.spin_state = qt.basis(2, 0)
            else:
                self.spin_state = qt.basis(2, 1)
            self.density_matrix = qt.ket2dm(self.spin_state)
            child = self.try_reproduce()
            self.Z_dipole_len = DIPOLE_LEN_INIT
            self.Z_compression = 0.0
            self.growth = 0.0
            self.dynamic_energy = 0.0
            self.potential_energy = 0.0
            self.energy_total = 0.0
            self.orbital_count = 0.0
            self.E_x = 0.0
            self.c_y = 0.0
            self.m_z = 0.0
            self.model_flip_count += 1
            self.flip_cycles.append(self.cycle_count)
            self.model_orientation = -self.model_orientation
            self.update_sensors()
            return child

        if self.energy_total >= E_MAX_QUANTUM:
            self.orient_Y = -self.orient_Y
            self.orient_Z = -self.orient_Z
            self.spin = -self.spin
            if self.spin > 0.0:
                self.spin_state = qt.basis(2, 0)
            else:
                self.spin_state = qt.basis(2, 1)
            child = self.try_reproduce()
            self.E_x = 0.0
            self.c_y = 0.0
            self.m_z = 0.0
            self.energy_total = 0.0
            self.delta = 0.0
            self.temperature = 0.0
            self.Z_dipole_len = Z_MAX
            self.Z_compression = 0.0
            self.Y_dipole_len = 6 * L_PL
            self.X_positive_arm = 3 * L_PL
            self.X_negative_arm = 3 * L_PL
            self.X_dipole_len = 6 * L_PL
            self.X_asymmetry = 0.0
            self.flicker_phase = 1.0 - self.flicker_phase
            self.model_flip_count += 1
            self.flip_cycles.append(self.cycle_count)
            self.model_orientation = -self.model_orientation
            self.calibrate_to_planck()
            self.update_sensors()
            return child

        return None

    def try_reproduce(self):
        if self.compare_diff(self.growth, GROWTH_THRESHOLD) < 0.0 and self.energy_total < E_MAX_QUANTUM:
            return None
        self.growth = 0.0
        self.detach_count += 1
        dV_repro = self.get_potential_diff_reproduction(self.charge, self.spin)
        child = PlanckQuantum(
            idx=self.detach_count,
            spin_dir=-self.spin,
            charge_sign=-self.charge,
            generation=self.generation + 1.0)
        child.E_x = self.E_x
        child.c_y = self.c_y
        child.m_z = self.m_z
        child.energy_total = self.energy_total
        child.dynamic_energy = self.dynamic_energy
        child.potential_energy = self.potential_energy
        child.orbital_count = self.orbital_count
        child.growth = self.growth
        child.Z_dipole_len = DIPOLE_LEN_INIT
        child.Z_compression = 0.0
        child.center_X = (self.center_X[0] + R_OUTER,
                          self.center_X[1] + R_OUTER,
                          self.center_X[2])
        child.center_Y = (self.center_Y[0] + R_OUTER,
                          self.center_Y[1] + R_OUTER,
                          self.center_Y[2])
        child.center_Z = (self.center_Z[0] + R_OUTER,
                          self.center_Z[1] + R_OUTER,
                          self.center_Z[2])
        child.sync_pos_from_centers()
        child.sensor = {
            'X+': L_PL, 'X-': L_PL,
            'Y+': L_PL, 'Y-': L_PL,
            'Z+': L_PL, 'Z-': L_PL
        }
        child.spin_state = self.spin_state.copy()
        child.density_matrix = self.density_matrix.copy()
        child.reproduction_potential_diff = dV_repro
        child.is_zero_state = (child.energy_total <= 0.0)
        self.children.append(child)
        return child

    def apply_external_sources(self, sources_X=None, sources_Y=None, sources_Z=None):
        self.stored_sources_X = sources_X
        self.stored_sources_Y = sources_Y
        self.stored_sources_Z = sources_Z
        self.feel_all_directions(sources_X, sources_Y, sources_Z)

    def calibrate_to_planck(self):
        N_X = self.X_dipole_len / L_PL if self.X_dipole_len > 0 else 0.0
        N_Y = self.Y_dipole_len / L_PL if self.Y_dipole_len > 0 else 0.0
        N_Z = self.Z_dipole_len / L_PL if self.Z_dipole_len > 0 else 0.0
        self.planck_energy_X = E_PLANCK / N_X if N_X > 0 else 0.0
        self.planck_energy_Y = E_PLANCK / N_Y if N_Y > 0 else 0.0
        self.planck_energy_Z = E_PLANCK / N_Z if N_Z > 0 else 0.0
        self.planck_voltage_X = V_PLANCK / N_X if N_X > 0 else 0.0
        self.planck_voltage_Y = V_PLANCK / N_Y if N_Y > 0 else 0.0
        self.planck_voltage_Z = V_PLANCK / N_Z if N_Z > 0 else 0.0
        self.planck_magnetic_Y = B_PLANCK / N_Y if N_Y > 0 else 0.0
        self.planck_emf_Y = EMF_PLANCK / N_Y if N_Y > 0 else 0.0
        self.E_times_N_X = self.planck_energy_X * N_X if N_X > 0 else 0.0
        self.E_times_N_Y = self.planck_energy_Y * N_Y if N_Y > 0 else 0.0
        self.E_times_N_Z = self.planck_energy_Z * N_Z if N_Z > 0 else 0.0
        self.planck_energy_total = (self.planck_energy_X +
                                     self.planck_energy_Y +
                                     self.planck_energy_Z)
        self.E_times_N_total = (self.E_times_N_X +
                                 self.E_times_N_Y +
                                 self.E_times_N_Z)
        self.n_points = self.E_times_N_total / E_PLANCK if E_PLANCK > 0 else 0.0
        if C_LIGHT_VAL > 0.0:
            self.planck_mass = self.planck_energy_total / (C_LIGHT_VAL ** 2)
        if K_B > 0.0:
            self.planck_temp = self.planck_energy_total / K_B
        self.is_cross = (self.Z_dipole_len <= L_PL)

    def reset(self):
        self.__init__(self.id, self.spin, self.charge, self.generation)

    def report(self):
        report = {
            'id': self.id,
            'cycle': self.cycle_count,
            'pos': (self.pos_X, self.pos_Y, self.pos_Z),
            'centers': {
                'X': self.center_X,
                'Y': self.center_Y,
                'Z': self.center_Z,
            },
            'centers_potential_diff': {
                'XY': self.get_potential_diff_XY(),
                'YZ': self.get_potential_diff_YZ(),
                'ZX': self.get_potential_diff_ZX(),
                'total': self.get_total_centers_potential_diff(),
            },
            'E_x': self.E_x,
            'c_y': self.c_y,
            'm_z': self.m_z,
            'energy_total': self.energy_total,
            'dynamic_energy': self.dynamic_energy,
            'potential_energy': self.potential_energy,
            'orbital_count': self.orbital_count,
            'delta': self.delta,
            'temperature': self.temperature,
            'pull': (self.pull_X(), self.pull_Y(), self.pull_Z()),
            'potential_diffs': {
                'X': self.get_potential_diff_X(),
                'Y': self.get_potential_diff_Y(),
                'Z': self.get_potential_diff_Z(),
                'total': self.get_total_potential_diff(),
                'ECM': self.get_potential_diff_ECM(),
                'reproduction': getattr(self, 'reproduction_potential_diff', 0.0),
                'same_charge': self.get_total_potential_diff_same_charge(self.chain_next) if self.chain_next else 0.0
            },
            'charge_potential_diff': self.get_charge_potential_diff(self.charge),
            'spin_potential_diff': self.get_spin_potential_diff(self.spin),
            'growth': self.growth,
            'detach_count': self.detach_count,
            'merged': self.merged,
            'spin': self.spin,
            'charge': self.charge,
            'model_flips': self.model_flip_count,
            'spin_flips': self.spin_flip_count,
            'spin_phase': self.spin_phase,
            'orientation': self.model_orientation,
            'Z_dipole_len': self.Z_dipole_len,
            'Z_compression': self.Z_compression,
            'spin_state': self.spin_state,
            'density_matrix': self.density_matrix,
            'energy_fraction': self.energy_fraction(),
            'is_zero_state': self.is_zero_state,
            'spin_freq': self.spin_frequency(),
            'z_pulse_freq': self.z_pulse_frequency(),
            'orbital_freq': self.orbital_frequency(),
            'electron_count': self.electron_count,
            'luminosity': self.luminosity,
            'Y_dipole_len': self.Y_dipole_len,
            'X_dipole_len': self.X_dipole_len,
            'X_asymmetry': self.X_asymmetry,
            'planck_energy_X': self.planck_energy_X,
            'planck_energy_Y': self.planck_energy_Y,
            'planck_energy_Z': self.planck_energy_Z,
            'planck_energy_total': self.planck_energy_total,
            'E_times_N_X': self.E_times_N_X,
            'E_times_N_Y': self.E_times_N_Y,
            'E_times_N_Z': self.E_times_N_Z,
            'E_times_N_total': self.E_times_N_total,
            'n_points': self.n_points,
            'magnetic_closed': self.magnetic_closed,
            'gravity_sign': self.gravity_sign,
            'gravitational_potential': self.planck_gravitational_potential,
            'Z_plus_center': self.Z_plus_center,
            'Z_minus_orbital': self.Z_minus_orbital,
            'Z_orbital_radius': self.Z_orbital_radius,
            'gravitational_wells': self.gravitational_wells,
            'is_cross': self.is_cross,
            'flicker_phase': self.flicker_phase,
        }
        if self.merged:
            report['merge_report'] = self.get_merge_report()
        if self.chain_linked:
            report['chain_report'] = self.get_chain_report()
        return report

    def planck_report(self):
        return {
            'id': self.id, 'cycle': self.cycle_count,
            'E_x_diff': self.E_x, 'c_y_diff': self.c_y, 'm_z_diff': self.m_z,
            'energy_diff': self.energy_total,
            'E_X_J': self.planck_energy_X,
            'E_Y_J': self.planck_energy_Y,
            'E_Z_J': self.planck_energy_Z,
            'E_total_J': self.planck_energy_total,
            'E_MAX_J': E_MAX_QUANTUM,
            'V_X_V': self.planck_voltage_X,
            'V_Y_V': self.planck_voltage_Y,
            'V_Z_V': self.planck_voltage_Z,
            'B_Y_T': self.planck_magnetic_Y,
            'EMF_Y_V': self.planck_emf_Y,
            'm_kg': self.planck_mass,
            'T_K': self.planck_temp,
            'E_PLANCK': E_PLANCK,
            'V_PLANCK': V_PLANCK,
            'B_PLANCK': B_PLANCK,
            'EMF_PLANCK': EMF_PLANCK,
            'E_X_over_E_P': self.planck_energy_X / E_PLANCK if E_PLANCK else 0.0,
            'E_total_over_E_P': self.planck_energy_total / E_PLANCK if E_PLANCK else 0.0,
            'E_total_over_E_MAX': self.planck_energy_total / E_MAX_QUANTUM if E_MAX_QUANTUM else 0.0,
            'magnetic_closed': self.magnetic_closed,
            'B_internal_T': self.planck_magnetic_internal,
            'B_external_T': self.planck_magnetic_external,
            'gravity_sign': self.gravity_sign,
            'gravitational_potential_J': self.planck_gravitational_potential,
            'E_times_N_X': self.E_times_N_X,
            'E_times_N_Y': self.E_times_N_Y,
            'E_times_N_Z': self.E_times_N_Z,
            'E_times_N_total': self.E_times_N_total,
            'n_points': self.n_points,
        }

    # ========================================================
    # НАМЁК НА СЛИЯНИЕ (без реализации PDM)
    # ========================================================

    def try_merge(self, other):
        """
        Намёк на слияние. В полной версии здесь создаётся PDM.
        В полусекретной версии — только проверка условий и заглушка.
        """
        dX = self.diff3D(self.center_X, other.center_X)
        dY = self.diff3D(self.center_Y, other.center_Y)
        dZ = self.diff3D(self.center_Z, other.center_Z)
        close_x = self.compare_diff(dX, MERGE_DISTANCE) <= 0.0
        close_y = self.compare_diff(dY, MERGE_DISTANCE) <= 0.0
        close_z = self.compare_diff(dZ, MERGE_DISTANCE) <= 0.0
        if not (close_x and close_y and close_z): return False
        if not (self.charge == -other.charge and self.spin == -other.spin): return False

        self.spin = -self.spin
        self.orient_Y = -self.orient_Y
        self.orient_Z = -self.orient_Z
        other.spin = -other.spin
        other.orient_Y = -other.orient_Y
        other.orient_Z = -other.orient_Z
        self.merge_orientation_Y = self.orient_Y
        self.merge_orientation_Z = self.orient_Z
        other.merge_orientation_Y = other.orient_Y
        other.merge_orientation_Z = other.orient_Z

        lock_Y = self.get_potential_diff_magnetic_lock_Y(other)
        lock_Z = self.get_potential_diff_magnetic_lock_Z(other)
        lock_formed = (lock_Y < GROWTH_THRESHOLD and lock_Z < GROWTH_THRESHOLD)
        if not lock_formed:
            self.spin = -self.spin
            self.orient_Y = -self.orient_Y
            self.orient_Z = -self.orient_Z
            other.spin = -other.spin
            other.orient_Y = -other.orient_Y
            other.orient_Z = -other.orient_Z
            return False

        self.merge_energy = self.get_dark_matter_energy(other)
        self.merge_potential_diff = self.get_potential_diff_dark_matter(other)
        self.merge_stage = 3
        other.merge_stage = 3
        self.partner = other
        other.partner = self
        self.merged = True
        other.merged = True
        self.merge_potential_diff += lock_Y + lock_Z
        other.merge_potential_diff = self.merge_potential_diff

        # НАМЁК: здесь в полной версии создаётся PDM.
        # В полусекретной версии — только заглушка.
        # self.annihilate_EM()
        # other.annihilate_EM()
        # gen.create_from_quanta(qA=self, qB=other, pos=center)

        return True

    def can_merge(self, other):
        dX = self.diff3D(self.center_X, other.center_X)
        dY = self.diff3D(self.center_Y, other.center_Y)
        dZ = self.diff3D(self.center_Z, other.center_Z)
        close_x = self.compare_diff(dX, MERGE_DISTANCE) <= 0.0
        close_y = self.compare_diff(dY, MERGE_DISTANCE) <= 0.0
        close_z = self.compare_diff(dZ, MERGE_DISTANCE) <= 0.0
        spatial_close = close_x and close_y and close_z
        charge_compatible = self.charge == -other.charge
        spin_compatible = self.spin == -other.spin
        energy_barrier = self.get_total_potential_diff_merge(other)
        barrier_ok = energy_barrier < GROWTH_THRESHOLD
        dark_matter_diff = self.get_potential_diff_dark_matter(other)
        dark_matter_ok = dark_matter_diff > 0
        lock_Y = self.get_potential_diff_magnetic_lock_Y(other)
        lock_Z = self.get_potential_diff_magnetic_lock_Z(other)
        lock_ok = (lock_Y < GROWTH_THRESHOLD and lock_Z < GROWTH_THRESHOLD)
        return (spatial_close and charge_compatible and spin_compatible and
                barrier_ok and dark_matter_ok and lock_ok)

    def get_dark_matter_energy(self, other):
        dark_matter_diff = self.get_potential_diff_dark_matter(other)
        total_energy = self.energy_total + other.energy_total
        return dark_matter_diff + total_energy

    def get_merge_report(self):
        return {
            'merge_energy': self.merge_energy,
            'merge_potential_diff': self.merge_potential_diff,
            'merge_stage': self.merge_stage,
            'charge_total': self.charge + (self.partner.charge if self.partner else 0),
            'spin_total': self.spin + (self.partner.spin if self.partner else 0),
            'position': (self.pos_X, self.pos_Y, self.pos_Z),
            'centers': (self.center_X, self.center_Y, self.center_Z),
            'partner_id': self.partner.id if self.partner else None,
            'merged': self.merged,
            'orientation_Y': self.merge_orientation_Y,
            'orientation_Z': self.merge_orientation_Z,
            'magnetic_lock_Y': self.get_potential_diff_magnetic_lock_Y(self.partner) if self.partner else 0,
            'magnetic_lock_Z': self.get_potential_diff_magnetic_lock_Z(self.partner) if self.partner else 0
        }

    # ========================================================
    # ЦЕПОЧКА (полная реализация)
    # ========================================================

    def try_chain_merge(self, other):
        dX = self.diff3D(self.center_X, other.center_X)
        dY = self.diff3D(self.center_Y, other.center_Y)
        dZ = self.diff3D(self.center_Z, other.center_Z)
        close_x = self.compare_diff(dX, CHAIN_MERGE_DISTANCE) <= 0.0
        close_y = self.compare_diff(dY, CHAIN_MERGE_DISTANCE) <= 0.0
        close_z = self.compare_diff(dZ, CHAIN_MERGE_DISTANCE) <= 0.0
        if not (close_x and close_y and close_z): return False
        connect_diff = self.get_total_potential_diff_chain_connect(other)
        chain_diff = self.get_total_potential_diff_chain(other)
        same_diff = self.get_total_potential_diff_same_charge(other)
        if not (connect_diff < GROWTH_THRESHOLD and
                chain_diff < GROWTH_THRESHOLD and
                same_diff < GROWTH_THRESHOLD):
            return False
        self.chain_next = other
        other.chain_prev = self
        self.chain_linked = True
        other.chain_linked = True
        self.chain_energy = self.get_chain_binding_energy(other)
        other.chain_energy = self.chain_energy
        self.chain_potential_diff = self.get_chain_potential_diff_binding(other)
        other.chain_potential_diff = self.chain_potential_diff
        if self.chain_prev is not None:
            self.chain_length = self.chain_prev.chain_length + 1
        else:
            self.chain_length = 1
        if other.chain_next is not None:
            other.chain_length = other.chain_next.chain_length + 1
        else:
            other.chain_length = 1
        self.update_chain_length()
        other.update_chain_length()
        return True

    def update_chain_length(self):
        current = self
        length = 1
        while current.chain_next is not None:
            current = current.chain_next
            length += 1
            current.chain_length = length
        current = self
        length = 1
        while current.chain_prev is not None:
            current = current.chain_prev
            length += 1
            current.chain_length = length

    def get_chain_report(self):
        return {
            'chain_linked': self.chain_linked,
            'chain_length': self.chain_length,
            'chain_energy': self.chain_energy,
            'chain_potential_diff': self.chain_potential_diff,
            'chain_next_id': self.chain_next.id if self.chain_next else None,
            'chain_prev_id': self.chain_prev.id if self.chain_prev else None,
            'same_charge_diff': self.get_total_potential_diff_same_charge(self.chain_next) if self.chain_next else 0.0,
            'position': (self.pos_X, self.pos_Y, self.pos_Z)
        }

    def can_chain_merge(self, other):
        dX = self.diff3D(self.center_X, other.center_X)
        dY = self.diff3D(self.center_Y, other.center_Y)
        dZ = self.diff3D(self.center_Z, other.center_Z)
        close_x = self.compare_diff(dX, CHAIN_MERGE_DISTANCE) <= 0.0
        close_y = self.compare_diff(dY, CHAIN_MERGE_DISTANCE) <= 0.0
        close_z = self.compare_diff(dZ, CHAIN_MERGE_DISTANCE) <= 0.0
        spatial_close = close_x and close_y and close_z
        connect_diff = self.get_total_potential_diff_chain_connect(other)
        connect_ok = connect_diff < GROWTH_THRESHOLD
        chain_diff = self.get_total_potential_diff_chain(other)
        chain_ok = chain_diff < GROWTH_THRESHOLD
        same_diff = self.get_total_potential_diff_same_charge(other)
        same_ok = same_diff < GROWTH_THRESHOLD
        not_in_chain = (not self.chain_linked or not other.chain_linked or
                        self.chain_next is not other or self.chain_prev is not other)
        return (spatial_close and connect_ok and chain_ok and same_ok and not_in_chain)

    # ========================================================
    # АВТОНОМНОСТЬ
    # ========================================================

    def scan_package_for_instances(self, package_name):
        instances = []
        if not package_name: return instances
        try:
            pkg = importlib.import_module(package_name)
        except Exception:
            return instances
        pkg_dir = os.path.dirname(pkg.__file__) if hasattr(pkg, '__file__') and pkg.__file__ else None
        if pkg_dir is None or not os.path.isdir(pkg_dir): return instances
        for fname in sorted(os.listdir(pkg_dir)):
            if not fname.endswith('.py'): continue
            if fname.startswith('__'): continue
            mod_name = fname[:-3]
            full_name = f"{package_name}.{mod_name}"
            try:
                mod = importlib.import_module(full_name)
                inst = getattr(mod, 'INSTANCE', None)
                if inst is not None: instances.append(inst)
            except Exception:
                continue
        return instances

    def scan_own_package(self):
        return self.scan_package_for_instances('quanta')

    def scan_dark_matter_package(self):
        return self.scan_package_for_instances('dark_matter')

    def autolink_neighbors(self):
        if not hasattr(self, 'neighbor_dm_left'): self.init_neighbors()
        dm_list = self.scan_dark_matter_package()
        q_list = self.scan_own_package()
        q_list_others = [q for q in q_list if q is not self and q.id != self.id]
        detected = self.detect_neighbors_by_field(dm_list)
        self.quanta_ref = q_list_others
        return {'dm_total': len(dm_list), 'q_total': len(q_list),
                'detected': detected,
                'left': self.neighbor_dm_left.id if self.neighbor_dm_left else None,
                'right': self.neighbor_dm_right.id if self.neighbor_dm_right else None}

    def init_neighbors(self):
        self.neighbor_dm_left = None
        self.neighbor_dm_right = None
        self.neighbor_detected = False
        self.neighbor_field_left = 0.0
        self.neighbor_field_right = 0.0

    def detect_neighbors_by_field(self, candidate_dms, detection_radius=None):
        if detection_radius is None:
            detection_radius = R_SENSOR * GROWTH_THRESHOLD
        self.neighbor_detected = False
        best_left = None
        best_right = None
        best_d_left = float('inf')
        best_d_right = float('inf')
        for dm in candidate_dms:
            if dm is None: continue
            dm_pos = (dm.pos[0], dm.pos[1], dm.pos[2]) if hasattr(dm, 'pos') else (0.0, 0.0, 0.0)
            d = self.diff3D((self.pos_X, self.pos_Y, self.pos_Z), dm_pos)
            if d > detection_radius: continue
            field_val = self.field_response_at_sensor(dm)
            if self.abs_diff(field_val) < L_PL * 1e-12: continue
            dy = dm_pos[1] - self.pos_Y
            if dy >= 0.0:
                if abs(dy) < best_d_right:
                    best_d_right = abs(dy)
                    best_right = dm
            else:
                if abs(dy) < best_d_left:
                    best_d_left = abs(dy)
                    best_left = dm
        self.neighbor_dm_left = best_left
        self.neighbor_dm_right = best_right
        self.neighbor_detected = (best_left is not None) or (best_right is not None)
        return self.neighbor_detected

    def field_amplitude_X(self): return self.E_x
    def field_amplitude_Y(self): return self.c_y
    def field_amplitude_Z(self): return self.m_z
    def field_phase_X(self):
        return 2.0 * math.pi * self.spin_frequency() * float(self.cycle_count)
    def field_phase_Y(self):
        return 2.0 * math.pi * self.spin_frequency() * float(self.cycle_count) + math.pi / 2.0
    def field_phase_Z(self):
        return 2.0 * math.pi * self.z_pulse_frequency() * float(self.cycle_count)
    def field_wave_number(self):
        lam = self.Z_dipole_len if self.Z_dipole_len > 0.0 else L_PL
        return 2.0 * math.pi / lam
    def field_response_X(self, distance):
        if distance <= 0.0: return self.field_amplitude_X()
        k = self.field_wave_number()
        phase = self.field_phase_X() - k * distance
        attenuation = 1.0 / (1.0 + distance / (GROWTH_THRESHOLD * L_PL))
        return self.field_amplitude_X() * math.cos(phase) * attenuation
    def field_response_Y(self, distance):
        if distance <= 0.0: return self.field_amplitude_Y()
        k = self.field_wave_number()
        phase = self.field_phase_Y() - k * distance
        attenuation = 1.0 / (1.0 + distance / (GROWTH_THRESHOLD * L_PL))
        return self.field_amplitude_Y() * math.cos(phase) * attenuation
    def field_response_Z(self, distance):
        if distance <= 0.0: return self.field_amplitude_Z()
        k = self.field_wave_number()
        phase = self.field_phase_Z() - k * distance
        attenuation = 1.0 / (1.0 + distance / (GROWTH_THRESHOLD * L_PL))
        return self.field_amplitude_Z() * math.cos(phase) * attenuation
    def field_response_at_sensor(self, other):
        if other is None: return 0.0
        other_pos = (other.pos[0], other.pos[1], other.pos[2]) if hasattr(other, 'pos') else (0.0, 0.0, 0.0)
        distance = self.diff3D((self.pos_X, self.pos_Y, self.pos_Z), other_pos)
        if hasattr(other, 'field_response_X'):
            return (other.field_response_X(distance) +
                    other.field_response_Y(distance) +
                    other.field_response_Z(distance))
        else:
            v = getattr(other, 'V', L_PL)
            attenuation = 1.0 / (1.0 + distance / (GROWTH_THRESHOLD * L_PL))
            return v * attenuation

    def collect_sources_from_neighbors(self):
        sources_X = []
        sources_Y = []
        sources_Z = []
        for dm in (self.neighbor_dm_left, self.neighbor_dm_right):
            if dm is None: continue
            dm_pos = (dm.pos[0], dm.pos[1], dm.pos[2]) if hasattr(dm, 'pos') else (0.0, 0.0, 0.0)
            v = getattr(dm, 'm_z', getattr(dm, 'V', L_PL))
            sources_Z.append((dm_pos, v))
        return sources_X, sources_Y, sources_Z

    def collect_sources(self):
        sX, sY, sZ = self.collect_sources_from_neighbors()
        return sX, sY, sZ

    def init_event_slots(self):
        self.last_born_child = None
        self.last_merged_partner = None
        self.last_chain_partner = None
        self.last_emitted_charge = None

    def tick(self):
        if not hasattr(self, 'neighbor_dm_left'): self.init_neighbors()
        if not hasattr(self, 'last_born_child'): self.init_event_slots()
        self.init_event_slots()
        if self.merged:
            return {'born': None, 'merged': self.partner, 'chain': None,
                    'emitted': None, 'cycle': self.cycle_count}
        was_merged = self.merged
        was_chain = self.chain_linked
        n_charged_before = len(self.emitted_charges)
        sources_X, sources_Y, sources_Z = self.collect_sources()
        if sources_X or sources_Y or sources_Z:
            self.apply_external_sources(sources_X, sources_Y, sources_Z)
        else:
            self.stored_sources_X = None
            self.stored_sources_Y = None
            self.stored_sources_Z = None
        child = self.move()
        if child is not None: self.last_born_child = child
        if (not was_merged) and self.merged: self.last_merged_partner = self.partner
        if (not was_chain) and self.chain_linked: self.last_chain_partner = self.chain_next
        if len(self.emitted_charges) > n_charged_before:
            self.last_emitted_charge = self.emitted_charges[-1]
        return {'born': self.last_born_child, 'merged': self.last_merged_partner,
                'chain': self.last_chain_partner, 'emitted': self.last_emitted_charge,
                'cycle': self.cycle_count}

    def tick_count(self, n):
        events = []
        for _ in range(n):
            if self.merged:
                events.append({'born': None, 'merged': self.partner,
                               'chain': None, 'emitted': None,
                               'cycle': self.cycle_count})
                break
            events.append(self.tick())
        return events

    def report_events(self):
        return {'id': self.id, 'cycle': self.cycle_count,
                'last_born_child': self.last_born_child.id if self.last_born_child else None,
                'last_merged_partner': self.last_merged_partner.id if self.last_merged_partner else None,
                'last_chain_partner': self.last_chain_partner.id if self.last_chain_partner else None,
                'last_emitted_charge': self.last_emitted_charge is not None,
                'merged': self.merged, 'chain_linked': self.chain_linked}

    def run(self, n_ticks, stop_on_merge=True, log_events=False):
        events_all = []
        for i in range(n_ticks):
            if self.merged and stop_on_merge:
                events_all.append({'tick': i, 'born': None, 'merged': self.partner,
                                   'chain': None, 'emitted': None,
                                   'cycle': self.cycle_count, 'stopped': True})
                break
            event = self.tick()
            event['tick'] = i
            event['stopped'] = False
            events_all.append(event)
            if log_events and (event['born'] or event['merged'] or event['chain'] or event['emitted']):
                print(f"[{self.id}] tick={i} cycle={self.cycle_count} "
                      f"born={event['born'].id if event['born'] else None} "
                      f"merged={event['merged'].id if event['merged'] else None} "
                      f"chain={event['chain'].id if event['chain'] else None} "
                      f"emitted={event['emitted'] is not None}")
        return events_all

    def autonomous_run(self, n_ticks=1000, autolink=True, log_events=False):
        if autolink:
            link_info = self.autolink_neighbors()
        else:
            link_info = {'detected': False, 'left': None, 'right': None}
        events = self.run(n_ticks=n_ticks, stop_on_merge=True, log_events=log_events)
        return {'id': self.id, 'uid': getattr(self, 'uid', None),
                'ticks_done': len(events), 'cycle_final': self.cycle_count,
                'energy_final': self.energy_total, 'merged': self.merged,
                'chain_linked': self.chain_linked, 'flips': self.model_flip_count,
                'link_info': link_info,
                'last_event': events[-1] if events else None,
                'report': self.report()}


INSTANCE = PlanckQuantum(idx="q_0000", spin_dir=1, charge_sign=1, generation=0)


def _autonomous_main():
    import argparse
    parser = argparse.ArgumentParser(description="Автономный квант v25")
    parser.add_argument('--ticks', type=int, default=1000)
    parser.add_argument('--no-autolink', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()
    q = PlanckQuantum(idx="cli", spin_dir=1, charge_sign=1, generation=0)
    result = q.autonomous_run(
        n_ticks=args.ticks,
        autolink=(not args.no_autolink),
        log_events=args.log)
    print("=" * 60)
    print(f"КВАНТ {result['id']}")
    print(f"  тактов:        {result['ticks_done']}")
    print(f"  цикл:          {result['cycle_final']}")
    print(f"  энергия:       {result['energy_final']:.6e}")
    print(f"  слился:        {result['merged']}")
    print(f"  в цепочке:     {result['chain_linked']}")
    print(f"  флипов:        {result['flips']}")
    print(f"  соседей:       {result['link_info']}")
    print("=" * 60)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Отчёт сохранён: {args.json}")


if __name__ == "__main__":
    _autonomous_main()
