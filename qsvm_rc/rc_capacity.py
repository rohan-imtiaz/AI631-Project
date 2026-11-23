"""Biaxial capacity of a rectangular RC section.

Uniaxial capacities come from strain compatibility with the ACI rectangular stress block,
and they are combined with the Bresler reciprocal load method:
    1 / Pni = 1 / Pnx + 1 / Pny - 1 / P0
Forces are returned in kN, lengths are in mm, stresses in MPa.
"""

import numpy as np

from . import config

STEEL_MODULUS_MPA = 200000
CONCRETE_CRUSHING_STRAIN = 0.003


def beta1_aci(fc_mpa):
    """ACI 318-19 stress block depth factor."""
    return float(np.clip(0.85 - 0.05 * (fc_mpa - 28) / 7, 0.65, 0.85))


def bar_coordinates(section):
    """x and y coordinates (mm) of every bar, measured from the bottom left corner."""
    cover = section["cover_to_centroid_mm"]
    x_positions = np.linspace(cover, section["width_x_mm"] - cover, section["bars_long_face"])
    y_positions = np.linspace(cover, section["depth_y_mm"] - cover, section["bars_short_face"])

    # Bars on the two long faces (all x positions), then the interior bars of the two short faces
    bars_x = np.concatenate([x_positions, x_positions,
                             np.full(len(y_positions) - 2, cover),
                             np.full(len(y_positions) - 2, section["width_x_mm"] - cover)])
    bars_y = np.concatenate([np.full(len(x_positions), cover),
                             np.full(len(x_positions), section["depth_y_mm"] - cover),
                             y_positions[1:-1], y_positions[1:-1]])
    return bars_x, bars_y


def section_constants(section):
    """Gross area, steel area, P0 and the Bresler validity limit (kN)."""
    gross_area = section["width_x_mm"] * section["depth_y_mm"]
    bars_x, _ = bar_coordinates(section)
    steel_area = len(bars_x) * section["bar_area_mm2"]
    fc = section["fc_mpa"]
    fy = section["fy_mpa"]
    p0_kn = (0.85 * fc * (gross_area - steel_area) + fy * steel_area) / 1000
    return {
        "number_of_bars": len(bars_x),
        "gross_area_mm2": gross_area,
        "steel_area_mm2": steel_area,
        "steel_ratio": steel_area / gross_area,
        "p0_kN": p0_kn,
        "phi_p0_kN": config.PHI * p0_kn,
        "bresler_limit_kN": 0.1 * fc * gross_area / 1000,
        "beta1": beta1_aci(fc),
    }


def uniaxial_interaction_curve(depth_mm, breadth_mm, layer_depths_mm, layer_areas_mm2, fc_mpa, fy_mpa,
                               subtract_displaced_concrete = True, number_of_points = 20000):
    """Pn and Mn (N, N mm) over a sweep of neutral axis depths.

    depth_mm is the dimension parallel to the eccentricity, layer_depths_mm are measured
    from the compression face. Moments are taken about the geometric centre, which is the
    plastic centroid for a doubly symmetric section.
    """
    beta1 = beta1_aci(fc_mpa)
    neutral_axis_depths = np.geomspace(0.02 * depth_mm, 200 * depth_mm, number_of_points)
    stress_block_depths = np.minimum(beta1 * neutral_axis_depths, depth_mm)
    concrete_force = 0.85 * fc_mpa * stress_block_depths * breadth_mm

    layer_depths = np.asarray(layer_depths_mm, dtype = float)[None, :]
    layer_areas = np.asarray(layer_areas_mm2, dtype = float)[None, :]
    strains = CONCRETE_CRUSHING_STRAIN * (neutral_axis_depths[:, None] - layer_depths) / neutral_axis_depths[:, None]
    stresses = np.clip(STEEL_MODULUS_MPA * strains, -fy_mpa, fy_mpa)
    if subtract_displaced_concrete:
        inside_block = layer_depths < stress_block_depths[:, None]
        stresses = np.where(inside_block, stresses - 0.85 * fc_mpa, stresses)
    steel_forces = stresses * layer_areas

    axial = concrete_force + steel_forces.sum(axis = 1)
    moment = (concrete_force * (depth_mm / 2 - stress_block_depths / 2)
              + (steel_forces * (depth_mm / 2 - layer_depths)).sum(axis = 1))
    return axial, moment


def capacity_at_eccentricity(eccentricities_mm, axial, moment, p0_newton):
    """Interpolate Pn (N) at the requested eccentricities along the compression branch."""
    positive = axial > 0
    curve_eccentricity = moment[positive] / axial[positive]
    curve_axial = axial[positive]
    order = np.argsort(curve_eccentricity)
    capacities = np.interp(np.asarray(eccentricities_mm, dtype = float),
                           curve_eccentricity[order], curve_axial[order])
    capacities = np.where(np.asarray(eccentricities_mm) <= 0, p0_newton, capacities)
    return capacities


def uniaxial_capacities(section, ex_values_mm, ey_values_mm, subtract_displaced_concrete = True):
    """Pny (ex acting alone) and Pnx (ey acting alone), both in kN."""
    constants = section_constants(section)
    p0_newton = constants["p0_kN"] * 1000
    bars_x, bars_y = bar_coordinates(section)
    area = section["bar_area_mm2"]

    # ex acts along x, so the depth of the section in this direction is width_x
    axial_x, moment_x = uniaxial_interaction_curve(section["width_x_mm"], section["depth_y_mm"],
                                                   bars_x, np.full(len(bars_x), area),
                                                   section["fc_mpa"], section["fy_mpa"],
                                                   subtract_displaced_concrete = subtract_displaced_concrete)
    pny = capacity_at_eccentricity(ex_values_mm, axial_x, moment_x, p0_newton) / 1000

    axial_y, moment_y = uniaxial_interaction_curve(section["depth_y_mm"], section["width_x_mm"],
                                                   bars_y, np.full(len(bars_y), area),
                                                   section["fc_mpa"], section["fy_mpa"],
                                                   subtract_displaced_concrete = subtract_displaced_concrete)
    pnx = capacity_at_eccentricity(ey_values_mm, axial_y, moment_y, p0_newton) / 1000
    return pny, pnx


def biaxial_capacity(section, ex_mm, ey_mm, subtract_displaced_concrete = True):
    """Bresler Pni and phi Pni (kN) for arrays of (ex, ey). Uniaxial values are computed once per unique eccentricity."""
    ex_mm = np.asarray(ex_mm, dtype = float)
    ey_mm = np.asarray(ey_mm, dtype = float)
    unique_ex, ex_inverse = np.unique(ex_mm, return_inverse = True)
    unique_ey, ey_inverse = np.unique(ey_mm, return_inverse = True)
    pny_unique, _ = uniaxial_capacities(section, unique_ex, np.zeros(1), subtract_displaced_concrete)
    _, pnx_unique = uniaxial_capacities(section, np.zeros(1), unique_ey, subtract_displaced_concrete)
    pny = pny_unique[ex_inverse].reshape(ex_mm.shape)
    pnx = pnx_unique[ey_inverse].reshape(ey_mm.shape)

    p0 = section_constants(section)["p0_kN"]
    pni = 1 / (1 / pnx + 1 / pny - 1 / p0)
    return {
        "pny_kN": pny,
        "pnx_kN": pnx,
        "pni_kN": pni,
        "phi_pni_kN": config.PHI * pni,
        "bresler_valid": pni >= section_constants(section)["bresler_limit_kN"],
    }


def eccentricity_grid(half_dimension_mm):
    """10 mm steps over the first half of the half dimension, 5 mm steps over the second half."""
    first_half = np.arange(0, half_dimension_mm / 2, 10)
    second_start = 5 * np.ceil(half_dimension_mm / 2 / 5)
    second_half = np.arange(second_start, half_dimension_mm + 1e-9, 5)
    return np.unique(np.round(np.concatenate([first_half, second_half]), 3))


def axial_load_grid(section, start_kn = 1000, step_kn = 250):
    """Pu from 1000 kN in 250 kN steps, up to the first step above phi P0."""
    phi_p0 = section_constants(section)["phi_p0_kN"]
    stop = step_kn * np.ceil(phi_p0 / step_kn)
    return np.arange(start_kn, stop + 1e-9, step_kn)


def generate_section_dataset(section, subtract_displaced_concrete = True):
    """Full quadrant dataset (Pu, ex, ey, label) with intermediate values, as a dict of arrays."""
    ex_values = eccentricity_grid(section["width_x_mm"] / 2)
    ey_values = eccentricity_grid(section["depth_y_mm"] / 2)
    pu_values = axial_load_grid(section)

    pu_grid, ex_grid, ey_grid = np.meshgrid(pu_values, ex_values, ey_values, indexing = "ij")
    pu_flat = pu_grid.ravel()
    ex_flat = ex_grid.ravel()
    ey_flat = ey_grid.ravel()
    capacity = biaxial_capacity(section, ex_flat, ey_flat, subtract_displaced_concrete)
    status = np.where(pu_flat <= capacity["phi_pni_kN"], "SAFE", "UNSAFE")
    return {
        "pu_kN": pu_flat,
        "ex_mm": ex_flat,
        "ey_mm": ey_flat,
        "pny_kN": capacity["pny_kN"],
        "pnx_kN": capacity["pnx_kN"],
        "pni_kN": capacity["pni_kN"],
        "phi_pni_kN": capacity["phi_pni_kN"],
        "bresler_valid": capacity["bresler_valid"],
        "status": status,
    }
