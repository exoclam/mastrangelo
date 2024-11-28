import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
from pyia import GaiaData
from tqdm import tqdm

import simulate_helpers
import seaborn as sns
import pandas as pd
import numpy as np

# Gala
import gala.dynamics as gd
import gala.potential as gp

from astropy.table import Table, join
path = '/Users/chrislam/Desktop/mastrangelo/'
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
#path = '/home/c.lam/blue/sculpting2/'
#berger = Table.read(path+'berger_kepler_stellar_fgk.csv')

megan = Table.read(path+'data/kepler_dr3_good.fits')
#megan = Table.read(path+'kepler_dr3_good.fits')

merged = join(berger, megan, keys='kepid')
merged.rename_column('parallax_2', 'parallax')

with coord.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = coord.Galactocentric()

sun_xyz = u.Quantity(
    [-galcen_frame.galcen_distance, 0 * u.kpc, galcen_frame.z_sun]  # x  # y  # z
)

sun_w0 = gd.PhaseSpacePosition(pos=sun_xyz, vel=galcen_frame.galcen_v_sun)

mw_potential = gp.MilkyWayPotential()

sun_orbit = mw_potential.integrate_orbit(sun_w0, dt=0.5 * u.Myr, t1=0, t2=12 * u.Gyr) # t2 = 16 Gyr

#star_gaia = GaiaData(merged)
star_gaia = GaiaData(megan)

star_gaia_c = star_gaia.get_skycoord()

star_galcen = star_gaia_c.transform_to(galcen_frame)

star_w0 = gd.PhaseSpacePosition(star_galcen.data)

"""
pos_zs = []
vel_zs = []
vel_rs = []
pos_rs = []
for i in tqdm(range(len(star_gaia))):
#for i in range(4000):
    star_orbit = mw_potential.integrate_orbit(star_w0[i], t=sun_orbit.t)
    cstar_orbit = star_orbit.cylindrical

    pos_r = cstar_orbit.rho.value
    try:
        pos_r = np.array(pos_r[~np.isnan(pos_r)])[-1] #drop NaNs
    except:
        continue
    if (pos_r > 8.44) or (pos_r < 8.24):
        continue

    vel_z = star_orbit.vel.d_z.value
    try:    
        vel_z = np.array(vel_z[~np.isnan(vel_z)])[-1] #drop NaNs
    except: # sometimes, I guess integrate_orbit fails? 
        continue
    vel_z = simulate_helpers.kpc_per_myr_to_km_per_s(vel_z, 1)

    pos_z = star_orbit.pos.z.value
    try:
        pos_z = np.array(pos_z[~np.isnan(pos_z)])[-1] #drop NaNs
        #pos_z = np.nanmedian(pos_z)
    except:
        continue

    vel_r = cstar_orbit.v_rho.value
    try:
        vel_r = np.array(vel_r[~np.isnan(vel_r)])[-1] #drop NaNs
        #vel_r = np.nanmedian(np.array(vel_r))
    except:
        continue
    vel_r = simulate_helpers.kpc_per_myr_to_km_per_s(vel_r, 1)

    #print("v_z: ", star_orbit.vel[0].d_z.to(u.km))
    #print("v_z: ", vel_z)
    #print("v_r: ", vel_r) 
    #print("pos_z: ", pos_z)

    pos_zs.append(pos_z)
    vel_zs.append(vel_z)
    vel_rs.append(vel_r)

#plt.scatter(pos_zs, vel_zs, c=vel_rs, s=10)
snail_df = pd.DataFrame({'pos_zs': pos_zs, 'vel_zs': vel_zs, 'vel_rs': vel_rs})
snail_df.to_csv(path+'galactic-occurrence/data/snail.csv', index=False)
"""
snail_df = pd.read_csv(path+'galactic-occurrence/data/snail.csv')

fig, ax = plt.subplots(figsize=(6,6)) 
bins_pos_zs = np.round(np.arange(-1, 1.02, 0.04), 2)
bins_vel_zs = np.arange(-60, 61, 2.)
snail_df['bins_pos_zs'] = pd.cut(snail_df['pos_zs'], bins_pos_zs)
snail_df['bins_vel_zs'] = pd.cut(snail_df['vel_zs'], bins_vel_zs)
print(snail_df)
piv = snail_df.groupby(['bins_pos_zs','bins_vel_zs'], observed=True)['vel_rs'].median().reset_index()
#piv = snail_df.groupby(['bins_pos_zs','bins_vel_zs']).median('vel_rs').reset_index()
print("piv group by: ", piv)
piv = piv.pivot(index="bins_vel_zs",columns="bins_pos_zs",values="vel_rs")
print("piv: ", piv)
g = sns.heatmap(piv, vmin=-10, vmax=10)#, yticklabels=bins_vel_zs, xticklabels=bins_pos_zs)
#g.set_xticks(np.arange(-1, 1.02, 0.04))
#g.set_xticklabels(np.arange(-1, 1.02, 0.04))
#g.set_yticks(np.arange(-60, 61, 2.))
#g.set_yticklabels(np.arange(-60, 61, 2.))
#plt.yticks(bins_vel_zs[::2]) # sample every other tick, for cleanness
#plt.scatter(snail_df['pos_zs'], snail_df['vel_zs'], c=snail_df['vel_rs'], s=10, vmin=-10, vmax=10)
#plt.contourf(snail_df['pos_zs'], snail_df['vel_zs'], c=snail_df['vel_rs'], vmin=-10, vmax=10)
plt.ylabel(r'$V_Z$ (km $s^{-1}$)')
plt.xlabel('Z (kpc)')
#plt.ylim([-60, 60])
#plt.xlim([-1, 1])
#plt.colorbar(label=r'$V_R$ (km $s^{-1}$)')
fig.tight_layout()
#plt.savefig(path+'galactic-occurrence/plots/snail.png', bbox_inches='tight')
plt.show()
quit()

#### Ma+ 2017 scale heights, for comparison
#berger_kepler = pd.read_csv(path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
merged_df = merged.to_pandas()

#berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
berger_kepler_temp = simulate_helpers.draw_asymmetrically(merged_df, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')
berger_kepler_temp = simulate_helpers.draw_galactic_heights(berger_kepler_temp)

plt.scatter(berger_kepler_temp['height']/1000, zmaxes, s=10, color='powderblue')

plt.xlabel('Ma+ 2017 and isochrone age-inferred heights')
plt.ylabel('Gala-calculated heights')
#path = '/home/c.lam/blue/'
plt.savefig(path+'galactic-occurrence/plots/ma_vs_gala_heights.png')
plt.show()