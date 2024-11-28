import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astropy.table import Table, join
from astroquery.gaia import Gaia

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simulate_helpers

# Suppress warnings. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')

from astroquery.gaia import Gaia
tables = Gaia.load_tables(only_names=True)

path = '/Users/chrislam/Desktop/mastrangelo/'

"""
megan = Table.read(path+'data/kepler_dr3_good.fits')
#merged = join(berger, megan, keys='kepid')
#merged.rename_column('parallax_2', 'parallax')
print(megan)
print(list(megan.columns))
quit()
### doing it programatically
from pyia import GaiaData

for row in tic_table:
    data = GaiaData.from_source_id(
        row["dr2_source_id"], source_id_dr="dr2", data_dr="dr3"
    )

quit()
"""

"""
### only do this once!
Gaia.login(user='clam03', password='Roger.That1') # i am the paragon of security
Gaia.ROW_LIMIT = 200 # 6500000
#query_text = "SELECT G.source_id, G.radial_velocity, G.radial_velocity_error, G.ra, G.ra_error, G.dec, G.dec_error, G.parallax, G.parallax_error, G.pmra, G.pmra_error, G.pmdec, G.pmdec_error, G.ra_dec_corr, G.ra_parallax_corr, G.ra_pmra_corr, G.ra_pmdec_corr, G.dec_parallax_corr, G.dec_pmra_corr, G.dec_pmdec_corr, G.parallax_pmra_corr, G.parallax_pmdec_corr, G.pmra_pmdec_corr FROM gaiadr2.gaia_source G WHERE G.radial_velocity IS NOT Null AND G.parallax_over_error>5"
query_text = "SELECT G.source_id, G.radial_velocity, G.radial_velocity_error, G.ra, G.ra_error, G.dec, G.dec_error, G.parallax, G.parallax_error, G.pmra, G.pmra_error, G.pmdec, G.pmdec_error, G.ra_dec_corr, G.ra_parallax_corr, G.ra_pmra_corr, G.ra_pmdec_corr, G.dec_parallax_corr, G.dec_pmra_corr, G.dec_pmdec_corr, G.parallax_pmra_corr, G.parallax_pmdec_corr, G.pmra_pmdec_corr FROM gaiadr3.gaia_source G WHERE G.radial_velocity IS NOT Null AND G.parallax_over_error>5"
job = Gaia.launch_job_async(query_text)
r = job.get_results()
r_df = r.to_pandas()
r_df.to_csv(path+'galactic-occurrence/data/astroquery_antoja_dr3.csv', index=False)
Gaia.logout()
quit()
"""

"""
### this is APW's script for Dax from the Astro Data Group Slack
query = "SELECT tic.dr2_source_id, dr3.* FROM gaiadr3.gaia_source as dr3" 
query += "JOIN gaiadr3.dr2_neighbourhood as xm ON dr3.source_id=xm.dr3_source_id"
query += "JOIN user_clam03.tic_table_name as tic ON tic.dr2_source_id=xm.dr2_source_id"
job = Gaia.launch_job_async(query)#, dump_to_file=True)
r = job.get_results()
print(r)
quit()
"""

""" 
# get Kepler field stars. run once! 
Gaia.ROW_LIMIT = 10 #300000
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
coord = SkyCoord(l=76.3, b=13.5, unit=(u.degree, u.degree), frame='galactic')
width = u.Quantity(np.sqrt(10), u.deg)
height = u.Quantity(np.sqrt(10), u.deg)
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
r_df = r.to_pandas()
print(list(r_df.columns))
quit()
r_df = r_df[['dist', 'solution_id', 'DESIGNATION', 'SOURCE_ID', 'l', 'b', 'ra', 'dec', 'ra_error', 'dec_error', 'distance_gspphot', 'distance_gspphot_lower', 'distance_gspphot_upper', 'radial_velocity', 'rv_amplitude_robust', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']]
print(r_df['SOURCE_ID'])
#r_df.to_csv(path+'galactic-occurrence/data/astroquery_kepler.csv', index=False)
quit()
"""

#"""
### Kepler hosts
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
megan = Table.read(path+'data/kepler_dr2_1arcsec.fits') # kepler_dr3_good.fits
merged_berger_bedell = join(berger, megan, keys='kepid') # 70K stars
merged_berger_bedell = merged_berger_bedell.to_pandas()

kois = pd.read_csv(path+'data/cumulative_2021.03.04_20.04.43.csv')
kois = kois.loc[kois.koi_disposition != 'FALSE POSITIVE']
positives_kepler = pd.merge(kois, merged_berger_bedell, how='inner', on='kepid') 
positives_kepler = positives_kepler.drop_duplicates(subset='kepid')
#"""

"""
### Kepler-Gaia only
r_df = pd.read_csv(path+'galactic-occurrence/data/astroquery_kepler.csv')
r_df = r_df.dropna(subset=['distance_gspphot', 'distance_gspphot_upper', 'distance_gspphot_lower', 'radial_velocity', 'pmra', 'pmdec', 'ra', 'dec'])
gaia_kepler = pd.merge(positives_kepler, r_df, left_on='source_id', right_on='SOURCE_ID')
gaia_kepler = gaia_kepler.drop_duplicates(subset='kepid')
#print(list(gaia_kepler.columns))
gaia_kepler['ra'] = gaia_kepler['ra_x']
gaia_kepler['dec'] = gaia_kepler['dec_x']
gaia_kepler['pmra'] = gaia_kepler['pmra_x']
gaia_kepler['pmdec'] = gaia_kepler['pmdec_x']
gaia_kepler['radial_velocity'] = gaia_kepler['radial_velocity_x']
gaia_kepler['distance_gspphot'] = gaia_kepler['distance_gspphot_x']
#print(gaia_kepler)
"""

#"""
### CREATE GREAT SNAIL IN THE SKY
# read in data; dropna; enrich
r_df = pd.read_csv(path+'galactic-occurrence/data/astroquery_antoja.csv')
#r_df = r_df.dropna(subset=['distance_gspphot', 'radial_velocity', 'pmra', 'pmdec', 'ra', 'dec'])
#r_df['distance_gspphot_err1'] = r_df['distance_gspphot_upper'] - r_df['distance_gspphot']
#r_df['distance_gspphot_err2'] = r_df['distance_gspphot'] - r_df['distance_gspphot_lower']
#r_df['distance'] = simulate_helpers.draw_asymmetrically(r_df, 'distance_gspphot', 'distance_gspphot_err1', 'distance_gspphot_err2', 'distance')

### this is for DR3
#icrs_coord = SkyCoord(ra=np.array(r_df['ra'])*u.mas, dec=np.array(r_df['dec'])*u.mas,
#                            distance=np.array(r_df['distance_gspphot'])*u.pc, pm_ra_cosdec=np.array(r_df['pmra']) * u.mas / u.yr, pm_dec=np.array(r_df['pmdec']) * u.mas / u.yr, 
#                            radial_velocity=np.array(r_df['radial_velocity']) * u.km / u.s, frame='icrs')

### this is for DR2
icrs_coord = SkyCoord(ra=np.array(r_df['ra'])*u.deg, dec=np.array(r_df['dec'])*u.deg,
                            distance=1./np.array(r_df['parallax']*u.mas) * u.kpc, pm_ra_cosdec=np.array(r_df['pmra']) * u.mas / u.yr, pm_dec=np.array(r_df['pmdec']) * u.mas / u.yr, 
                            radial_velocity=np.array(r_df['radial_velocity']) * u.km / u.s, frame='icrs')

# Convert to Galactocentric coordinates
galactocentric = icrs_coord.transform_to('galactocentric')
cylindrical = galactocentric.represent_as('cylindrical')

# Antojas+ 2018 only kept the Solar annulus among Gaia stars
r_df['galz'] = galactocentric.z.value
r_df['galvz'] = galactocentric.v_z.value
r_df['galvr'] = galactocentric.radial_velocity.value
r_df['r'] = cylindrical.rho.value
r_df['galx'] = galactocentric.x.value

### Turn this off when I don't want to join with the Bedell planet-host Kepler-Gaia cross-match
r_df_kepler = positives_kepler.merge(r_df, left_on='source_id', right_on='SOURCE_ID') 
r_df_kepler['pmra'] = r_df_kepler['pmra_x']
r_df_kepler['pmdec'] = r_df_kepler['pmdec_x']
r_df_kepler['radial_velocity'] = r_df_kepler['radial_velocity_x']

#snail_df = r_df
snail_df_kepler = r_df_kepler

#plt.hist(np.abs(r_df['r']), bins=10)
#plt.hist(np.abs(r_df['galx']), bins=10)
#plt.show()
#quit()
snail_df = r_df.loc[(np.abs(r_df['galx']) < 8.44) & (np.abs(r_df['galx']) > 8.24)]
#plt.hist(r_df['r'])
#plt.show()
#quit()

fig, ax = plt.subplots(figsize=(6,6)) 

bins_galz = np.round(np.arange(-1, 1.02, 0.02), 2)
bins_galvz = np.arange(-60, 61, 1.)
snail_df['bins_galz'] = pd.cut(snail_df['galz'], bins_galz)
snail_df['bins_galvz'] = pd.cut(snail_df['galvz'], bins_galvz)
snail_df_kepler['bins_galz'] = pd.cut(snail_df_kepler['galz'], bins_galz)
snail_df_kepler['bins_galvz'] = pd.cut(snail_df_kepler['galvz'], bins_galvz)

piv = snail_df.groupby(['bins_galz','bins_galvz'], observed=False)['galvr'].median().reset_index()
piv_kepler = snail_df_kepler.groupby(['bins_galz','bins_galvz'], observed=False)['galvr'].median().reset_index()

piv = piv.pivot_table(index="bins_galvz",columns="bins_galz",values="galvr", dropna=False, sort=True)
piv_kepler = piv_kepler.pivot_table(index="bins_galvz",columns="bins_galz",values="galvr", dropna=False, sort=True)

piv_diff = piv - piv_kepler

#sns.heatmap(piv, annot=True, fmt=".1f")#, vmin=-10, vmax=10)
g = sns.heatmap(piv_kepler, vmin=-10, vmax=10, cmap='RdBu_r')
#g = sns.heatmap(piv_diff, cmap='RdBu_r', cbar_kws={'label': '$V_R$ (km $s^{-1}$)'})

plt.ylabel(r'$V_Z$ (km $s^{-1}$)')
plt.xlabel('Z (kpc)')
#plt.ylim([-60, 60])
#plt.xlim([-1, 1])
#plt.colorbar(label=r'$V_R$ (km $s^{-1}$)')
fig.tight_layout()
plt.savefig(path+'galactic-occurrence/plots/snail_kepler.png')
plt.show()
#"""

### CREATE A COMMON SNAIL
#from galpy.potential import MWPotential2014
