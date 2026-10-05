# -*- coding: utf-8 -*-
"""weather.py - weather data for the SolarStill model.

Three ways to obtain the hourly weather of a site:

1. fetch_nasa_power(lat, lon, date)      : hourly data from the NASA POWER database (free, worldwide, no account).
2. hourly_from_monthly(...)              : estimate from monthly DNI / GHI averages (solar atlas), when there is no internet.
3. the 'Meteo' sheet of the Excel file   : measurements entered by hand (default behaviour of the model).

apply_weather(ss, weather) then gives the weather to a SolarStill object, instead of the 'Meteo' sheet.

Author: Fatima Belmehdi.
"""
import json
import math
import datetime
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMETERS = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS2M,QV2M,PS"


# ----------------------------------------------------------------------------------------------------------------------
# Solar geometry
# ----------------------------------------------------------------------------------------------------------------------
def declination(n):
    """Solar declination in degrees for the day of the year n (Cooper)."""
    return 23.45 * math.sin(math.radians(360 / 365 * (284 + n)))


def cos_zenith(lat, delta, hour_angle):
    """Cosine of the solar zenith angle. Angles in degrees."""
    phi, d, h = map(math.radians, (lat, delta, hour_angle))
    return math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(h)


def cos_incidence(lat, delta, tilt, surface_azimuth, hour_angle):
    """
    Cosine of the angle of incidence of the sun on a tilted surface (Duffie and Beckman).
    lat, delta, tilt, hour_angle in degrees. surface_azimuth: 0 = facing the equator (south in the
    northern hemisphere), negative = east, positive = west.
    """
    phi, d, b, g, h = map(math.radians, (lat, delta, tilt, surface_azimuth, hour_angle))
    return (math.sin(d) * math.sin(phi) * math.cos(b)
            - math.sin(d) * math.cos(phi) * math.sin(b) * math.cos(g)
            + math.cos(d) * math.cos(phi) * math.cos(b) * math.cos(h)
            + math.cos(d) * math.sin(phi) * math.sin(b) * math.cos(g) * math.cos(h)
            + math.cos(d) * math.sin(b) * math.sin(g) * math.sin(h))


def tilted_irradiance(ghi, dni, dhi, lat, n, solar_hour, tilt, surface_azimuth=0.0, albedo=0.2):
    """
    Irradiance on a tilted surface in W/m2 (isotropic sky model):
    beam on the surface + diffuse from the sky + reflection from the ground.
    solar_hour: local solar time in hours (12 = solar noon).
    """
    delta = declination(n)
    hour_angle = 15.0 * (solar_hour - 12.0)
    if cos_zenith(lat, delta, hour_angle) <= 0:      # sun below the horizon
        return 0.0
    cos_i = max(cos_incidence(lat, delta, tilt, surface_azimuth, hour_angle), 0.0)
    b = math.radians(tilt)
    return max(dni * cos_i + dhi * (1 + math.cos(b)) / 2 + ghi * albedo * (1 - math.cos(b)) / 2, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# 1. NASA POWER
# ----------------------------------------------------------------------------------------------------------------------
def _to_w_m2(values, unit):
    """Convert an hourly solar quantity to a mean irradiance in W/m2, according to the unit given by the API."""
    u = (unit or "").replace(" ", "").lower()
    if u.startswith("mj"):            # MJ/m2 per hour
        return values * 1e6 / 3600.0
    if u.startswith("kw"):            # kW-hr/m2 per hour
        return values * 1000.0
    return values                      # W/m2 or Wh/m2 per hour: same number


def parse_nasa_power(payload, date, tilt=0.0, surface_azimuth=0.0, albedo=0.2, lat=None):
    """Turn the JSON answer of NASA POWER into a weather table (one row per hour of the day)."""
    par = payload["properties"]["parameter"]
    units = {k: v.get("units", "") for k, v in payload.get("parameters", {}).items()}
    if lat is None:
        lat = payload["geometry"]["coordinates"][1]
    day = date.strftime("%Y%m%d")
    n = date.timetuple().tm_yday
    rows = []
    for hour in range(24):
        key = f"{day}{hour:02d}"
        if key not in par["T2M"]:
            continue
        get = lambda name: float(par[name][key])
        ghi, dni, dhi = get("ALLSKY_SFC_SW_DWN"), get("ALLSKY_SFC_SW_DNI"), get("ALLSKY_SFC_SW_DIFF")
        if min(ghi, dni, dhi, get("T2M")) <= -990:       # -999 = missing value
            continue
        ghi = _to_w_m2(ghi, units.get("ALLSKY_SFC_SW_DWN"))
        dni = _to_w_m2(dni, units.get("ALLSKY_SFC_SW_DNI"))
        dhi = _to_w_m2(dhi, units.get("ALLSKY_SFC_SW_DIFF"))
        ig = ghi if tilt == 0 else tilted_irradiance(ghi, dni, dhi, lat, n, hour + 0.5, tilt, surface_azimuth, albedo)
        rows.append({
            "t": hour,                       # local solar time, hours
            "Ta": get("T2M"),                # ambient temperature, °C
            "Ig": max(ig, 0.0),              # irradiance on the cover, W/m2
            "GHI": ghi, "DNI": dni, "DHI": dhi,
            "V": get("WS2M") * 3.6,          # wind speed at 2 m, km/h
            "H": get("QV2M") / 1000.0,       # specific humidity, kg/kg
            "P": get("PS") / 1000.0,         # surface pressure, MPa
        })
    if not rows:
        raise ValueError("NASA POWER returned no valid data for " + day)
    return pd.DataFrame(rows)


def fetch_nasa_power(lat, lon, date, tilt=0.0, surface_azimuth=0.0, albedo=0.2, timeout=60):
    """
    Hourly weather of one day at (lat, lon) from NASA POWER.
    date: datetime.date or 'YYYY-MM-DD'. tilt: tilt of the cover in degrees (0 = horizontal).
    Returns a table with the columns t, Ta, Ig, V, H, P (same quantities as the 'Meteo' sheet).
    Data are available from 2001 to a few days before today.
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    day = date.strftime("%Y%m%d")
    query = urllib.parse.urlencode({
        "parameters": NASA_PARAMETERS, "community": "RE", "longitude": lon, "latitude": lat,
        "start": day, "end": day, "format": "JSON", "time-standard": "LST"})
    with urllib.request.urlopen(NASA_POWER_URL + "?" + query, timeout=timeout) as answer:
        payload = json.loads(answer.read().decode("utf-8"))
    return parse_nasa_power(payload, date, tilt, surface_azimuth, albedo, lat)


# ----------------------------------------------------------------------------------------------------------------------
# 2. Estimate from monthly averages (method of radiation.py)
# ----------------------------------------------------------------------------------------------------------------------
def hourly_from_monthly(lat, date, tilt, dni_month, ghi_month, days_in_month, gti_month=None,
                        sunrise=7, sunset=19, surface_azimuth=0.0, albedo=0.17):
    """
    Hourly irradiance on a tilted surface estimated from monthly totals (kWh/m2 per month), as given by a solar atlas.
    The monthly mean day is spread over the hours with a sine distribution between sunrise and sunset.
    gti_month: monthly total on the tilted plane, if known; the result is then scaled to match it.
    Returns a table with the columns t (hour) and Ig (W/m2).
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    n = date.timetuple().tm_yday
    hours = list(range(sunrise, sunset + 1))
    shape = [max(0.0, math.sin(math.pi * (h - sunrise) / (sunset - sunrise))) for h in hours]
    total = sum(shape)
    dni_day = dni_month * 1000.0 / days_in_month      # Wh/m2 per day
    ghi_day = ghi_month * 1000.0 / days_in_month
    irr = []
    for h, s in zip(hours, shape):
        f = s / total
        ghi, dni = ghi_day * f, dni_day * f
        delta = declination(n)
        hour_angle = 15.0 * (h - 12)
        cz = max(cos_zenith(lat, delta, hour_angle), 0.0)
        dhi = max(ghi - dni * cz, 0.0)
        irr.append(tilted_irradiance(ghi, dni, dhi, lat, n, h, tilt, surface_azimuth, albedo))
    if gti_month:
        estimated = sum(irr) * days_in_month / 1000.0
        if estimated > 0:
            irr = [i * gti_month / estimated for i in irr]
    return pd.DataFrame({"t": hours, "Ig": irr})


# ----------------------------------------------------------------------------------------------------------------------
# Give the weather to the model
# ----------------------------------------------------------------------------------------------------------------------
def apply_weather(ss, weather, start=None, end=None):
    """
    Replace the weather of a SolarStill object (read from the 'Meteo' sheet) by a weather table.
    weather: table with the columns t (hours), Ta (°C), Ig (W/m2) and, optionally, V (km/h), H (kg/kg), P (MPa).
    start, end: first and last hour to simulate (for example 7 and 19).
    Call it after creating the object and before ss_performances().
    """
    w = weather.copy()
    if start is not None:
        w = w[w["t"] >= start]
    if end is not None:
        w = w[w["t"] <= end]
    w = w.reset_index(drop=True)
    if len(w) < 2:
        raise ValueError("The weather table needs at least two hours")
    ss.ta = w["Ta"] + ss.kelv
    ss.ig = w["Ig"]
    if "V" in w:
        ss.v = float(np.mean(w["V"])) / ss.conv
    if "H" in w:
        ss.w = float(w["H"].iloc[0])     # specific humidity of the ambient air, kg/kg
    if "P" in w:
        ss.p = float(w["P"].iloc[0])
    ss.tf = np.array(w["t"], dtype=float)
    ss.t = np.linspace(min(ss.tf), max(ss.tf), len(ss.tf))
    ss.y = np.array(ss.ta).reshape(len(ss.ta), )
    ss.tamb = interp1d(ss.t, ss.y)
    ss.y1 = np.array(ss.ig).reshape(len(ss.ig), )
    ss.ig1 = interp1d(ss.t, ss.y1)
    return ss
