import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
from pyia import GaiaData
from tqdm import tqdm

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

sun_orbit = mw_potential.integrate_orbit(sun_w0, dt=0.5 * u.Myr, t1=0, t2=16 * u.Gyr)

star_gaia = GaiaData(merged)

star_gaia_c = star_gaia.get_skycoord()

star_galcen = star_gaia_c.transform_to(galcen_frame)

star_w0 = gd.PhaseSpacePosition(star_galcen.data)


pos_zs = []
vel_zs = []
pos_rs = []
vel_rs = []
for i in tqdm(range(len(star_gaia))):
#for i in range(4):
    star_orbit = mw_potential.integrate_orbit(star_w0[i], t=sun_orbit.t)
    vel_z = star_orbit.vel[0].d_z.value
    pos_z = star_orbit.pos[0].z.value
    cstar_orbit = star_orbit.cylindrical
    print("v_r: ", cstar_orbit[0].vel)
    quit()
    vel_r = cstar_orbit[0].vel[0] 
    pos_zs.append(pos_z)
    vel_zs.append(vel_z)
    vel_rs.append(vel_r)
    print(vel_z, vel_r)
    quit()

    pos_r = star_orbit.spherical.distance.value

    pos_rs.append(pos_r)

quit()

#### Ma+ 2017 scale heights, for comparison
import simulate_helpers
import pandas as pd

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